"""
BLE Indoor Localization Server
Receives RSSI from ESP32, runs 6 localization algorithms in real-time,
broadcasts results to the web dashboard via WebSocket.

Usage:
    python server.py --port COM5          # Windows 
    python server.py --port /dev/ttyUSB0  # Linux 
    python server.py --udp                # UDP
    python server.py --replay data/fingerprint_log.csv  # replay for ML training
    python server.py --port COM5 --fingerprints data/fingerprints.csv
Requirements:
    pip install pyserial websockets numpy scikit-learn filterpy torch flask
"""

import asyncio
import json
import math
import time
import argparse
import threading
import queue
import serial
import socket
import csv
import os
import sys
import threading
import numpy as np
from collections import deque
from datetime import datetime

import websockets
from flask import Flask, send_from_directory

# CONFIG — match your room and anchor positions

ROOM_X = 1.0   
ROOM_Y = 0.5   # in meters

# Anchor positions [x, y] in meters
ANCHORS = np.array([
    [0.5,  0.5],   
    [0.0, 0.0], 
    [1, 0],  
], dtype=float)

NUM_ANCHORS = len(ANCHORS)

# Path loss model, calibrate for each room. 
N     = 3    # path loss exponent  
C     = -75.0  # RSSI at 1 meter  
# C = average RSSI_measured when distance = 1m exactly
 

# Noise model, general values
NOISE_STD   = 2.0
SHADOW_STD  = 4.0

# UDP settings
UDP_PORT = 5005

# WebSocket settings
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# Flask static server
FLASK_PORT = 8080

# Shared state info

latest_data = {
    "rssi": [-100.0] * NUM_ANCHORS,
    "seen": [0] * NUM_ANCHORS,
    "positions": {
        "trilateration": [5.0, 5.0],
        "weighted":      [5.0, 5.0],
        "kalman":        [5.0, 5.0],
        "knn":           [5.0, 5.0],
        "mlp":           [5.0, 5.0],
        "cnn":           [5.0, 5.0],
    },
    "errors": {},
    "timestamp": 0,
    "true_pos": None,
}

path_history = {k: deque(maxlen=200) for k in latest_data["positions"]}
true_path_history = deque(maxlen=200)

ws_clients = set()
data_queue = queue.Queue()

 
# Path Loss utilities

def rssi_to_distance(rssi, n=N, c=C):
    """Convert RSSI (dBm) → estimated distance (m)"""
    rssi = np.clip(rssi, -120, -1)
    d = 10.0 ** ((c - rssi) / (10.0 * n))
    return max(d, 0.3)

def distance_to_rssi(d, n=N, c=C):
    return -10 * n * math.log10(max(d, 0.3)) + c

# Trilateration (Least squares)
def trilateration(distances):
    """Standard least-squares trilateration"""
    x1, y1 = ANCHORS[0]
    d1 = distances[0]

    A = []
    b = []
    for i in range(1, NUM_ANCHORS):
        xi, yi = ANCHORS[i]
        di = distances[i]
        A.append([2*(xi - x1), 2*(yi - y1)])
        b.append(d1**2 - di**2 - x1**2 + xi**2 - y1**2 + yi**2)

    A = np.array(A)
    b = np.array(b)
    try:
        pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return np.clip(pos, [0, 0], [ROOM_X, ROOM_Y]).tolist()
    except:
        return [5.0, 5.0]

 
# Algorithm 2: Weighted Trilateration

def weighted_trilateration(distances):
    # Weighted least-squares — closer anchors get higher weight"""
    # noise increases with distance
    x1, y1 = ANCHORS[0]
    d1 = distances[0]

    A = []
    b = []
    w = []
    for i in range(1, NUM_ANCHORS):
        xi, yi = ANCHORS[i]
        di = distances[i]
        A.append([2*(xi - x1), 2*(yi - y1)])
        b.append(d1**2 - di**2 - x1**2 + xi**2 - y1**2 + yi**2)
        w.append(1.0 / (di**2 + 1e-6))

    A = np.array(A)
    b = np.array(b)
    W = np.diag(w)
    try:
        AtWA = A.T @ W @ A
        AtWb = A.T @ W @ b
        pos = np.linalg.solve(AtWA, AtWb)
        return np.clip(pos, [0, 0], [ROOM_X, ROOM_Y]).tolist()
    except:
        return trilateration(distances)

 
# Kalman Filter

class KalmanFilter2D:
    """
    State: [x, y, vx, vy]
    Measurement: [x, y] from weighted trilateration
    """
    def __init__(self):
        dt = 1.0
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.eye(4) * 0.05   # process noise (tune: higher = follows measurement more)
        self.R = np.eye(2) * 1.5    # measurement noise (tune: higher = smoother but laggier)
        self.P = np.eye(4) * 2.0
        self.x = np.array([5.0, 5.0, 0.0, 0.0])

    def update(self, measurement):
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        z = np.array(measurement)
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        return np.clip(self.x[:2], [0, 0], [ROOM_X, ROOM_Y]).tolist()

kalman = KalmanFilter2D()

# Fingerprint Database (for kNN / MLP / CNN)

class FingerprintDB:
    def __init__(self):
        self.X = None  # shape (N, num_anchors)
        self.Y = None  # shape (N, 2)
        self.loaded = False
        self.mlp = None
        self.cnn = None
        self.scaler_X = None
        self.scaler_Y = None

    def load_csv(self, path):
        """
        CSV format (generated by calibration tool or fingerprint_collector.py):
        x,y,rssi0,rssi1,rssi2
        1.0,1.0,-65.2,-70.1,-68.4
        ...
        """
        data = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row['x'])
                y = float(row['y'])
                rssis = [float(row[f'rssi{i}']) for i in range(NUM_ANCHORS)]
                data.append([x, y] + rssis)
        data = np.array(data)
        self.Y = data[:, :2]
        self.X = data[:, 2:]
        self.loaded = True
        print(f"[FingerprintDB] Loaded {len(data)} samples from {path}")
        return self

    def generate_synthetic(self, n_per_point=30, grid_step=0.5):
        """
        Generate synthetic fingerprints using the path loss model.
        Use this BEFORE you collect real data — replace with real data for better accuracy.
        """
        print("[FingerprintDB] Generating synthetic fingerprints...")
        xs = np.arange(0, ROOM_X + grid_step, grid_step)
        ys = np.arange(0, ROOM_Y + grid_step, grid_step)
        X_list = []
        Y_list = []
        for x in xs:
            for y in ys:
                pos = np.array([x, y])
                for _ in range(n_per_point):
                    row = []
                    for a in ANCHORS:
                        d = max(np.linalg.norm(pos - a), 0.5)
                        rssi = -10 * N * math.log10(d) + C \
                               + np.random.normal(0, SHADOW_STD) \
                               + np.random.normal(0, NOISE_STD)
                        row.append(rssi)
                    X_list.append(row)
                    Y_list.append([x, y])
        self.X = np.array(X_list, dtype=float)
        self.Y = np.array(Y_list, dtype=float)
        self.loaded = True
        print(f"[FingerprintDB] Generated {len(self.X)} synthetic samples")

    def train_models(self):
        """Train MLP and CNN on the fingerprint database"""
        if not self.loaded:
            print("[FingerprintDB] No data loaded!")
            return

        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import train_test_split

        print("[FingerprintDB] Training models...")

        self.scaler_X = StandardScaler()
        Xn = self.scaler_X.fit_transform(self.X)

        X_tr, X_te, Y_tr, Y_te = train_test_split(Xn, self.Y, test_size=0.1)

        # MLP
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            max_iter=500,
            learning_rate_init=0.001,
            verbose=False,
            early_stopping=True,
            n_iter_no_change=20,
        )
        self.mlp.fit(X_tr, Y_tr)
        mlp_err = np.mean(np.sqrt(np.sum((self.mlp.predict(X_te) - Y_te)**2, axis=1)))
        print(f"[MLP] Test mean error: {mlp_err:.3f} m")

        # CNN (PyTorch 1D conv over anchors)
        try:
            self._train_cnn(Xn, self.Y)
        except Exception as e:
            print(f"[CNN] Training failed: {e}. CNN will use MLP fallback.")
            self.cnn = None

        print("[FingerprintDB] Training complete!")

    def _train_cnn(self, Xn, Y):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        class CNN1D(nn.Module):
            def __init__(self, n_anchors):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=2, padding=1),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.Conv1d(16, 32, kernel_size=2, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                )
                # Compute flattened size
                dummy = torch.zeros(1, 1, n_anchors)
                out = self.conv(dummy)
                flat = out.view(1, -1).shape[1]
                self.fc = nn.Sequential(
                    nn.Linear(flat, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2),
                )

            def forward(self, x):
                x = x.unsqueeze(1)  # (B, 1, n_anchors)
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNN1D(NUM_ANCHORS).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        Xt = torch.FloatTensor(Xn).to(device)
        Yt = torch.FloatTensor(Y).to(device)
        ds = TensorDataset(Xt, Yt)
        dl = DataLoader(ds, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(60):
            for xb, yb in dl:
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch+1) % 20 == 0:
                print(f"  [CNN] Epoch {epoch+1}/60 loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            test_pred = model(Xt[:100]).cpu().numpy()
            test_true = Y[:100]
            err = np.mean(np.sqrt(np.sum((test_pred - test_true)**2, axis=1)))
        print(f"[CNN] Test sample mean error: {err:.3f} m")

        self.cnn = model
        self.cnn_device = device
        print("[CNN] Training complete!")

    def predict_knn(self, rssi_vec, k=5):
        if not self.loaded:
            return [5.0, 5.0]
        dists = np.sqrt(np.sum((self.X - rssi_vec)**2, axis=1))
        idx = np.argsort(dists)[:k]
        w = 1.0 / (dists[idx] + 1e-6)
        pos = np.average(self.Y[idx], weights=w, axis=0)
        return np.clip(pos, [0, 0], [ROOM_X, ROOM_Y]).tolist()

    def predict_mlp(self, rssi_vec):
        if self.mlp is None:
            return [5.0, 5.0]
        xn = self.scaler_X.transform(rssi_vec.reshape(1, -1))
        pos = self.mlp.predict(xn)[0]
        return np.clip(pos, [0, 0], [ROOM_X, ROOM_Y]).tolist()

    def predict_cnn(self, rssi_vec):
        if self.cnn is None:
            return self.predict_mlp(rssi_vec)
        import torch
        xn = self.scaler_X.transform(rssi_vec.reshape(1, -1))
        xt = torch.FloatTensor(xn).to(self.cnn_device)
        with torch.no_grad():
            pos = self.cnn(xt).cpu().numpy()[0]
        return np.clip(pos, [0, 0], [ROOM_X, ROOM_Y]).tolist()

    def save_models(self, path="models/"):
        import pickle, torch
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "mlp.pkl"), "wb") as f:
            pickle.dump((self.mlp, self.scaler_X), f)
        if self.cnn:
            torch.save(self.cnn.state_dict(), os.path.join(path, "cnn.pth"))
        print(f"[FingerprintDB] Models saved to {path}")

    def load_models(self, path="models/"):
        import pickle
        mlp_path = os.path.join(path, "mlp.pkl")
        if os.path.exists(mlp_path):
            with open(mlp_path, "rb") as f:
                self.mlp, self.scaler_X = pickle.load(f)
            print("[FingerprintDB] MLP loaded from disk")
        cnn_path = os.path.join(path, "cnn.pth")
        if os.path.exists(cnn_path):
            try:
                import torch
                from server import FingerprintDB
                # Rebuild same CNN arch and load weights
                # (will be skipped if torch not available)
                print("[CNN] cnn.pth found — load manually if needed")
            except:
                pass

# Main Localization Engine
class LocalizationEngine:
    def __init__(self, fpdb: FingerprintDB):
        self.fpdb = fpdb
        self.kf = KalmanFilter2D()
        self.step = 0

    def process(self, rssi_vec):
        """
        rssi_vec: np.array of shape (NUM_ANCHORS,)
        Returns dict of positions keyed by method name
        """
        self.step += 1

        # Distance estimation from RSSI
        distances = np.array([rssi_to_distance(r) for r in rssi_vec])

        # 1. Trilateration
        pos_tri = trilateration(distances)

        # 2. Weighted Trilateration
        pos_wt = weighted_trilateration(distances)

        # 3. Kalman Filter (fed by weighted trilateration)
        pos_kf = self.kf.update(pos_wt)

        # 4. kNN Fingerprinting
        pos_knn = self.fpdb.predict_knn(rssi_vec)

        # 5. MLP Fingerprinting
        pos_mlp = self.fpdb.predict_mlp(rssi_vec)

        # 6. CNN Fingerprinting
        pos_cnn = self.fpdb.predict_cnn(rssi_vec)

        return {
            "trilateration": pos_tri,
            "weighted":      pos_wt,
            "kalman":        pos_kf,
            "knn":           pos_knn,
            "mlp":           pos_mlp,
            "cnn":           pos_cnn,
        }


# Serial Reader

def serial_reader(port, baud=115200):
    print(f"[Serial] Opening {port} at {baud}...")
    ser = serial.Serial(port, baud, timeout=2)
    print(f"[Serial] Connected!")
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('{'):
                data = json.loads(line)
                data_queue.put(data)
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"[Serial] Error: {e}")
            time.sleep(1)


# UDP Reader

def udp_reader():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    print(f"[UDP] Listening on port {UDP_PORT}...")
    while True:
        try:
            data_bytes, addr = sock.recvfrom(1024)
            data = json.loads(data_bytes.decode('utf-8'))
            data_queue.put(data)
        except Exception as e:
            print(f"[UDP] Error: {e}")


# WebSocket Server

async def ws_handler(websocket):
    ws_clients.add(websocket)
    print(f"[WS] Client connected: {websocket.remote_address}")
    try:
        # Send current state immediately
        await websocket.send(json.dumps(latest_data))
        async for _ in websocket:
            pass
    except:
        pass
    finally:
        ws_clients.discard(websocket)
        print(f"[WS] Client disconnected")

async def ws_broadcast(msg):
    global ws_clients
    if ws_clients:
        dead = set()
        for ws in ws_clients:
            try:
                await ws.send(msg)
            except:
                dead.add(ws)
        ws_clients -= dead

# Main Processing Loop
async def processing_loop(engine: LocalizationEngine):
    global latest_data
    print("[Engine] Processing loop started")
    while True:
        try:
            # Non-blocking queue check
            try:
                raw = data_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            rssi = np.array(raw.get("rssi", [-100] * NUM_ANCHORS), dtype=float)
            seen = raw.get("seen", [1] * NUM_ANCHORS)

            # Replace unseen anchors with last smooth or very low value
            for i in range(NUM_ANCHORS):
                if not seen[i]:
                    rssi[i] = max(rssi[i], -100)

            # Run all algorithms
            positions = engine.process(rssi)

            # Update history
            for method, pos in positions.items():
                path_history[method].append(pos)

            # Build broadcast payload
            latest_data = {
                "rssi": rssi.tolist(),
                "distances": [round(rssi_to_distance(r), 2) for r in rssi],
                "seen": seen,
                "positions": positions,
                "history": {k: list(v) for k, v in path_history.items()},
                "timestamp": raw.get("t", int(time.time() * 1000)),
                "step": engine.step,
                "anchors": ANCHORS.tolist(),
                "room": [ROOM_X, ROOM_Y],
            }

            await ws_broadcast(json.dumps(latest_data))

        except Exception as e:
            print(f"[Engine] Error: {e}")
            await asyncio.sleep(0.1)

# Flask Static Server for Dashboard
app = Flask(__name__, static_folder='../dashboard')

@app.route('/')
def index():
    return send_from_directory('../dashboard', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../dashboard', path)

def run_flask():
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False)


# Fingerprint Collector Mode

def collect_fingerprints(port, out_file="data/fingerprints.csv", samples_per_point=30):
    """
    Interactive fingerprint collection.
    Stand at a known position, type x,y → collect RSSI samples → move.
    """
    print("\n=== FINGERPRINT COLLECTION MODE ===")
    print(f"Saving to: {out_file}")
    print("Type 'x,y' to set position, ENTER to collect samples, 'q' to quit\n")

    ser = serial.Serial(port, 115200, timeout=2)
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    existing = os.path.exists(out_file)
    with open(out_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not existing:
            writer.writerow(['x', 'y'] + [f'rssi{i}' for i in range(NUM_ANCHORS)])

        current_pos = None
        while True:
            cmd = input("Position (x,y) or ENTER to collect or q: ").strip()
            if cmd == 'q':
                break
            elif ',' in cmd:
                parts = cmd.split(',')
                current_pos = (float(parts[0]), float(parts[1]))
                print(f"Position set to {current_pos}")
            elif cmd == '' and current_pos:
                print(f"Collecting {samples_per_point} samples at {current_pos}...")
                collected = 0
                while collected < samples_per_point:
                    try:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if line.startswith('{'):
                            data = json.loads(line)
                            rssi = data.get('rssi', [-100]*NUM_ANCHORS)
                            writer.writerow(list(current_pos) + rssi)
                            f.flush()
                            collected += 1
                            print(f"  [{collected}/{samples_per_point}] RSSI: {rssi}")
                    except:
                        pass
                print(f"✓ Collected {samples_per_point} samples at {current_pos}")
            else:
                print("Enter 'x,y' first")

    print(f"\nFingerprint collection done! File: {out_file}")


# Entry Point
async def main(args):
    # Setup fingerprint DB
    fpdb = FingerprintDB()
    if args.fingerprints and os.path.exists(args.fingerprints):
        fpdb.load_csv(args.fingerprints)
        fpdb.train_models()
        fpdb.save_models()
    else:
        print("[Server] No fingerprint CSV found — using SYNTHETIC data for ML models.")
        print("[Server] For better accuracy: run with --collect to gather real fingerprints first.")
        fpdb.generate_synthetic()        
        t = threading.Thread(target=fpdb.train_models, daemon=True)
        t.start()

    engine = LocalizationEngine(fpdb)

    # Start data reader
    if args.udp:
        t = threading.Thread(target=udp_reader, daemon=True)
        t.start()
    elif args.port:
        t = threading.Thread(target=serial_reader, args=(args.port,), daemon=True)
        t.start()
    else:
        print("ERROR: specify --port COMx or --udp")
        sys.exit(1)

    # Start Flask
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"[Flask] Dashboard at http://localhost:{FLASK_PORT}")

    # Start WebSocket server + processing loop
    ws_server = await websockets.serve(ws_handler, WS_HOST, WS_PORT)
    print(f"[WS] WebSocket server at ws://localhost:{WS_PORT}")

    await asyncio.gather(
        processing_loop(engine),
        ws_server.wait_closed(),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLE Localization Server")
    parser.add_argument('--port',         type=str, help='Serial port (e.g. COM5 or /dev/ttyUSB0)')
    parser.add_argument('--udp',          action='store_true', help='Use UDP instead of serial')
    parser.add_argument('--fingerprints', type=str, default='data/fingerprints.csv',
                        help='Path to fingerprint CSV file')
    parser.add_argument('--collect',      action='store_true', help='Run fingerprint collection mode')
    args = parser.parse_args()

    if args.collect:
        collect_fingerprints(args.port)
    else:
        asyncio.run(main(args))
