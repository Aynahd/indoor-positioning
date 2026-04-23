# BLE Indoor Localization  
 
## PROJECT STRUCTURE

```
ble_localization/
├── esp32/
│   └── ble_scanner.ino           
├── server/
│   ├── server.py                 
│   ├── collect_fingerprints.py   
│   └── requirements.txt
└── dashboard/
    └── index.html                
```

---

## STEP 1  PHONE SETUP (nRF Connect)

On each of the 3 phones:

1. Install nRF Connect for Mobile 
2. Open, tap the Advertiser tab (bottom)
3. Tap and to create a new advertising packet
4. Set Display Name:
   *(must match exactly — case-sensitive)*
5. Under Advertising Data, add:
   - `Complete Local Name` ← this broadcasts the name
   - `Tx Power Level`      ← helps with RSSI compensation
6. Under Options:
   -  Connectable  
   -  Scannable  
7. Set Tx Power: `1 dBm` (as in your screenshot — consistent across phones)
8. Set Interval: `100` (= 250ms — fast enough for localization)
9. Tap OK to start advertising
 

### Anchor Placement
Place phones at measured corners of your room:
```
Phone3 at top-center
─────────────────────
│                   │
│                   │
│                   │
Phone1 ─────── Phone2

 Use a tape measure. Record exact positions in meters.
 Edit ANCHORS in server.py to match
 Place phones at consistent heights (~1m off the ground on shelves/stands)



## STEP 2 ESP32 SETUP

### Required Libraries (Arduino IDE)
Install via Library Manager:
- `ArduinoJson` by Benoit Blanchon (v7.x)
- ESP32 BLE Arduino (built-in with ESP32 board package)

### Board Setup
- Board: `ESP32 Dev Module` (or your specific variant)
- Partition Scheme: `Default` or `No OTA (Large APP)`
- Upload Speed: `115200`

### Flash
1. Open `esp32/ble_scanner.ino`
2. Edit if using WiFi (SSID, password, server IP) — or keep SERIAL mode
3. Flash via USB
4. Open Serial Monitor at 115200 to verify JSON output:
   ```
   {"t":12345,"rssi":[-65.2,-70.1,-68.4],"seen":[1,1,1]}
   ```

---

## STEP 3 CALIBRATE PATH LOSS (Important for accuracy!)

The constant `C` in server.py is the RSSI at exactly 1 meter.

**To calibrate:**
1. Place phone and ESP32 exactly 1.00 meter apart
2. Watch Serial Monitor output
3. Average 20 readings of RSSI for each phone
4. Set `C = average_value` in server.py (typically -45 to -60 dBm)
5. Adjust `N` (path loss exponent):
   - Open space/office: N = 2.0–2.5
   - Cluttered/furniture: N = 3.0–3.5
   - Many walls/obstacles: N = 3.5–4.0

---

## STEP 4 COLLECT FINGERPRINTS

This step trains kNN, MLP, and CNN on real RSSI data.
**Skip for first run** — server will use synthetic data.

```bash
cd server
python collect_fingerprints.py --port COM5   # Windows
python collect_fingerprints.py --port /dev/ttyUSB0  # Linux/Mac
```

Walk the room in a grid every 0.5–1.0 meters:
- Stand still at each point
- Type `x,y` coordinates (e.g., `2.5,3.0`)
- Wait for 40 samples to collect
- Move to next point

**How many points?** For a 10×10m room at 1m grid = ~121 points (~5000 samples).
At 0.5m grid = ~441 points (~18,000 samples) — much better accuracy but takes ~1 hour.

- Hold phones at consistent height
- Same person walking (body affects BLE differently)
- Collect multiple times at different times of day (WiFi/people affect RSSI)
- More samples per point = less noisy fingerprint

---

## STEP 5 — INSTALL PYTHON DEPENDENCIES

```bash
cd server
pip install -r requirements.txt
```

---

## STEP 6 — RUN THE SERVER

```bash
cd server

# Serial mode (USB cable):
python server.py --port COM5                          # Windows
python server.py --port /dev/ttyUSB0                 # Linux

# With real fingerprints:
python server.py --port COM5 --fingerprints data/fingerprints.csv

# UDP mode (WiFi):
python server.py --udp
```

Then open your browser: **http://localhost:8080**

---

## TUNING GUIDE

### Kalman Filter (server.py)
```python
self.Q = np.eye(4) * 0.05   # Process noise
# ↑ Higher = filter reacts faster to movement, but noisier
# ↓ Lower  = filter is smoother but lags behind fast movement

self.R = np.eye(2) * 1.5    # Measurement noise
# ↑ Higher = trusts measurement less (smoother path)
# ↓ Lower  = trusts measurement more (jittery but responsive)
```

### MLP Architecture
```python
hidden_layer_sizes=(64, 32, 16)  # in server.py
# More layers/neurons = better for complex rooms
# Add dropout if overfitting: use MLPClassifier with different params
```

### kNN
- k=5 works well for 30+ samples per grid point
- Increase k to 7–10 if RSSI is very noisy

---

## EXPECTED ACCURACY 

| Method         | Typical Error |
|----------------|---------------|
| Trilateration  | 1.5–3.0 m     |
| Weighted       | 1.2–2.5 m     |
| Kalman Filter  | 0.8–1.8 m     |
| kNN (real FP)  | 0.5–1.2 m     |
| MLP (real FP)  | 0.4–1.0 m     |
| CNN (real FP)  | 0.4–1.0 m     |

With synthetic fingerprints, ML models will perform like trilateration.
Real fingerprints improve accuracy by 30–60%.

---

## TROUBLESHOOTING

**ESP32 can't see phones:**
- Check the Display Name in nRF Connect matches exactly
- Ensure advertising is running (green play button in nRF Connect)
- Try moving phones closer first to verify detection

**RSSI is -100 for one anchor:**
- The phone may have gone to sleep — increase screen timeout
- Check nRF Connect is still advertising (not auto-paused)

**Dashboard not loading:**
- Make sure server.py is running
- Check port 8080 and 8765 aren't blocked by firewall

**JSON output looks wrong on Serial:**
- Make sure ArduinoJson library version ≥ 6

**ML models not training:**
- Install PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- CNN will fall back to MLP if PyTorch fails
