"""
Fingerprint Collector
=====================
Standalone script to walk your room and collect real RSSI fingerprints.
This gives the ML models (kNN/MLP/CNN) real training data.

Usage:
    python collect_fingerprints.py --port COM5
    python collect_fingerprints.py --port /dev/ttyUSB0

Output: data/fingerprints.csv
"""

import serial
import json
import csv
import os
import time
import argparse
import numpy as np

NUM_ANCHORS = 3
OUT_FILE = "data/fingerprints.csv"
SAMPLES_PER_POINT = 40   # collect 40 readings per grid point

def collect(port, baud=115200):
    os.makedirs("data", exist_ok=True)
    print("=" * 55)
    print("  BLE Fingerprint Collector")
    print("=" * 55)
    print(f"Output: {OUT_FILE}")
    print(f"Samples per point: {SAMPLES_PER_POINT}")
    print()
    print("Instructions:")
    print("  1. Stand at a known location in the room")
    print("  2. Type the x,y coords and press ENTER")
    print("  3. Stay still while samples are collected")
    print("  4. Move to next point and repeat")
    print("  5. Aim for a grid every 0.5m across the whole room")
    print("  6. Type 'q' to quit")
    print()

    existing = os.path.exists(OUT_FILE)
    f = open(OUT_FILE, 'a', newline='')
    writer = csv.writer(f)
    if not existing:
        writer.writerow(['x', 'y'] + [f'rssi{i}' for i in range(NUM_ANCHORS)])
        f.flush()

    ser = serial.Serial(port, baud, timeout=3)
    time.sleep(1)
    print(f"Serial connected: {port}")

    total_points = 0
    total_samples = 0

    while True:
        cmd = input("\nEnter position (x,y) or 'q': ").strip()
        if cmd.lower() == 'q':
            break

        try:
            x, y = [float(v.strip()) for v in cmd.split(',')]
        except:
            print("  Format: x,y  e.g.  2.5,3.0")
            continue

        print(f"  Collecting at ({x}, {y}) — stay still!")
        
        buffer = []
        while len(buffer) < SAMPLES_PER_POINT:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line.startswith('{'):
                    continue
                data = json.loads(line)
                rssi = data.get('rssi', [-100]*NUM_ANCHORS)
                seen = data.get('seen', [1]*NUM_ANCHORS)
                
                # Only accept if all anchors are seen
                if all(seen):
                    buffer.append(rssi)
                    n = len(buffer)
                    bar = '█' * n + '░' * (SAMPLES_PER_POINT - n)
                    print(f"\r  [{bar}] {n}/{SAMPLES_PER_POINT}  RSSI: {[round(r,1) for r in rssi]}", end='', flush=True)
            except:
                pass

        print()  # newline after progress bar

        # Write all samples
        for rssi in buffer:
            writer.writerow([x, y] + rssi)
        f.flush()

        total_points += 1
        total_samples += len(buffer)
        print(f"  ✓ Saved. Total: {total_points} points, {total_samples} samples")

        # Show running stats
        avg_rssi = np.mean(buffer, axis=0)
        std_rssi = np.std(buffer, axis=0)
        print(f"  Mean RSSI: {[round(r,1) for r in avg_rssi]}")
        print(f"  Std  RSSI: {[round(s,1) for s in std_rssi]}")

    f.close()
    print(f"\nDone! {total_samples} total samples in {OUT_FILE}")
    print("Now run: python server.py --port COM5 --fingerprints data/fingerprints.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True, help='Serial port')
    args = parser.parse_args()
    collect(args.port)
