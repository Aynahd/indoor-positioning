
 /* BLE Indoor Localization - ESP32 Scanner
 
  Replace the anchor names below the exact phone's bluetooth name
 */

#include <BLEDevice.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>
#include <ArduinoJson.h>


// CONFIG — EDIT THESE TO MATCH YOUR PHONES

const char* ANCHOR_NAMES[] = {
  "Galaxy F15 5G",   
  "Unnati's M14",    
  "vivo Y56 5G"      
};
const int NUM_ANCHORS = 3;

const int   SCAN_SECS = 3;
const float ALPHA     = 0.4f;

//define states

BLEScan* pBLEScan;
float rssi_smooth[3] = {-100.0f, -100.0f, -100.0f}; // init to minimum values
float rssi_raw[3]    = {-100.0f, -100.0f, -100.0f};
bool  seen[3]        = {false, false, false}; // not seen in the first scan 
bool  first_scan     = true;


// BLE Callback

class ScanCallback : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice dev) override {
    String name    = dev.haveName() ? String(dev.getName().c_str()) : String("(unnamed)");
    String address = String(dev.getAddress().toString().c_str());
    int    rssi    = dev.getRSSI();

    Serial.printf("  >> '%s'  [%s]  RSSI:%d\n", name.c_str(), address.c_str(), rssi);

    for (int i = 0; i < NUM_ANCHORS; i++) {
      if (name == String(ANCHOR_NAMES[i])) {
        rssi_raw[i] = (float)rssi;
        seen[i]     = true;
        Serial.printf("  MATCHED anchor %d!\n", i);
      }
    }
  }
};


// Setup

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n[BLE-TAG] Starting...");
  Serial.println("[BLE-TAG] Looking for:");
  for (int i = 0; i < NUM_ANCHORS; i++)
    Serial.printf("  Anchor %d: '%s'\n", i, ANCHOR_NAMES[i]);

  BLEDevice::init("ESP32-LOC");
  delay(500);

  pBLEScan = BLEDevice::getScan();
  pBLEScan->setAdvertisedDeviceCallbacks(new ScanCallback(), true);
  pBLEScan->setActiveScan(true);
  pBLEScan->setInterval(100);
  pBLEScan->setWindow(99);

  Serial.println("[BLE-TAG] Ready!");
}

// Loop

void loop() {
  for (int i = 0; i < NUM_ANCHORS; i++) seen[i] = false;

  Serial.println("\n[SCAN] Scanning...");
  pBLEScan->start(SCAN_SECS, false);
  pBLEScan->clearResults();

  Serial.printf("[SCAN] Done. Seen: [%d,%d,%d]\n", seen[0], seen[1], seen[2]);

  for (int i = 0; i < NUM_ANCHORS; i++) {
    if (seen[i]) {
      rssi_smooth[i] = first_scan
        ? rssi_raw[i]
        : ALPHA * rssi_raw[i] + (1.0f - ALPHA) * rssi_smooth[i];
    }
  }
  first_scan = false;

  StaticJsonDocument<200> doc;
  doc["t"] = millis();
  JsonArray rssiArr = doc.createNestedArray("rssi");
  JsonArray seenArr = doc.createNestedArray("seen");
  for (int i = 0; i < NUM_ANCHORS; i++) {
    rssiArr.add((float)((int)(rssi_smooth[i] * 10)) / 10.0f);
    seenArr.add(seen[i] ? 1 : 0);
  }
  String json;
  serializeJson(doc, json);
  Serial.println(json);
}
