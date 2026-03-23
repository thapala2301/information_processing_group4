# Information_Processing_Group4

# Facial-Recognition Based IoT Attendance System
 
**Information Processing — Final Team Project**
 
Archit Kumar · Archit Bhansali · Shashank Vinoo · Marcus Ikpase · Divyaa Nagasundhar · Thanus Palakumar · Anu Sandilyan
 
---
 
## Overview
 
An IoT-based automated attendance system for a university lecture hall. Two PYNQ-Z1 FPGA nodes work in tandem: Node 1 performs real-time face detection and recognition, reporting attendance to a central AWS cloud server. Node 2 receives the access decision and drives a physical output layer consisting of an RGB LED and a servo motor simulating a door lock mechanism.
 
---
 
## Repository Structure
 
```
.
├── imgProc/                        # Node 1 — FPGA camera input and face detection
│   ├── faceDetection.ipynb         # HLS accelerator, face detection, MQTT publish
│   ├── test_with_accel.ipynb       # Testing notebook for accelerator verification
│   ├── haarcascade_frontalface_default.xml
│   ├── recreate_project.tcl        
│   ├── dependencies.txt
│   ├── accel_downscale/            # HLS accelerator source
│   └── hdmi_input/                 # HDMI input IP source
│
├── Face_Attendance/                # AWS cloud backend and face recognition
│   ├── match_face.py               # 128-D embedding matcher
│   ├── encodings_cache.pkl         # Cached face embeddings
│   ├── known_faces/                # Reference face images
│   ├── already_checked_images/     # Processed image store
│   └── README.md
│
├── output_side/                    # Node 2 — physical output node
│   ├── output_node.py              # PYNQ-Z1 MQTT subscriber and state dispatcher
│   └── Pi_Controller.py           # Raspberry Pi serial listener, GPIO + PWM driver
│
└── README.md
```
 
---
 
## System Architecture
 
The pipeline is divided into four stages:
 
1. **Camera Input (PYNQ-Z1 Node 1)** — HLS accelerator downscales the HDMI stream from 1920×1080 to 480×270 in the FPGA fabric. The ARM processor runs OpenCV Haar cascade face detection on the downscaled frame, maps bounding boxes back to full resolution, and crops the face from the original RGB frame.
 
2. **Face Recognition (AWS EC2)** — Cropped face images are published to AWS IoT Core over MQTT. The EC2 server computes a 128-D dlib embedding and performs Euclidean nearest-neighbour matching against a registered database. GRANT or DENY is published back to the nodes.
 
3. **Attendance Logging (AWS)** — Events are logged to DynamoDB. Unknown faces are uploaded to S3. A Flask dashboard gives lecturers a live attendance view.
 
4. **Physical Output (PYNQ-Z1 Node 2 + Raspberry Pi)** — Node 2 subscribes to the decision topic and dispatches to one of four access states, communicating with a Raspberry Pi over USB serial to drive the RGB LED via GPIO and the SG90 servo via PWM.
 
---
 
## Hardware Requirements
 
### Node 1
- PYNQ-Z1 board
- HDMI input source (camera or video feed)
 
### Node 2
- PYNQ-Z1 board
- Raspberry Pi (any model with GPIO)
- RGB LED (connected to GPIO pins 17, 27, 22)
- SG90 micro servo motor (connected to GPIO pin 18)
- USB serial cable between PYNQ and Raspberry Pi
 
### Cloud
- AWS EC2 t3.micro instance (Ubuntu 24.04, eu-west-2)
- AWS IoT Core, DynamoDB, S3
 
---
 
## Setup and Running
 
### Node 1 — FPGA Camera Input
 
1. Copy the contents of `imgProc/` to `/home/xilinx/jupyter_notebooks/IP PRJ/` on the PYNQ.
2. Place Node 1 certificates in `/home/xilinx/jupyter_notebooks/IP PRJ/certs/`:
```
AmazonRootCA1.pem
node1-certificate.pem.crt
node1-private.pem.key
```
3. Install dependencies:
```bash
pip install -r dependencies.txt
```
4. Connect HDMI source to the PYNQ board.
5. Open `faceDetection.ipynb` in Jupyter and run all cells.
 
To recreate the Vivado project from source:
```bash
vivado -source recreate_project.tcl
```
 
### Node 2 — Output Node (PYNQ)
 
1. Copy the contents of `output_side/` to `/home/xilinx/jupyter_notebooks/IP PRJ/` on the second PYNQ.
2. Place Node 2 certificates in `/home/xilinx/jupyter_notebooks/IP PRJ/certs/`:
```
AmazonRootCA1.pem
node2-certificate.pem.crt
node2-private.pem.key
```
3. Install dependencies:
```bash
pip install paho-mqtt pyserial
```
4. Connect Raspberry Pi via USB serial.
5. Run:
```bash
python output_node.py
```
 
### Raspberry Pi — GPIO Controller
 
1. Copy `Pi_Controller.py` to the Raspberry Pi.
2. Install dependencies:
```bash
pip install pyserial RPi.GPIO
```
3. Wire RGB LED to GPIO pins 17 (red), 27 (green), 22 (blue) and servo to GPIO pin 18. Adjust pin numbers in `Pi_Controller.py` if your wiring differs.
4. Run:
```bash
python Pi_Controller.py
```
 
### AWS Cloud Backend
 
1. Launch an EC2 t3.micro instance (Ubuntu 24.04, eu-west-2).
2. Install dependencies and deploy `match_face.py` and the attendance server.
3. Configure AWS IoT Core with per-device certificates for each node.
4. Add reference face images to `known_faces/` and run `match_face.py` to generate `encodings_cache.pkl`.
 
---
 
## Access States
 
| State   | LED Colour   | LED Pattern  | Servo                    |
|---------|-------------|--------------|--------------------------|
| Grant   | Green        | Steady 3s    | Opens for 3s then closes |
| Deny    | Red          | Steady 3s    | Remains closed           |
| Pending | Amber/Yellow | Slow pulse   | Remains closed           |
| Alarm   | Red          | Rapid flash  | Remains closed           |
 
---
 
## Security Features
 
- **Anti-tailgate** — 30-second cooldown per named person prevents duplicate attendance records.
- **Repeated denial response** — alarm triggers after 3 consecutive unknown rejections, or 5 denials within a 120-second rolling window. Resets on grant.
- **Graceful degradation** — if the Raspberry Pi serial link is unavailable, the system enters LED-only mode automatically. All events continue to log locally.
- **Heartbeat monitor** — background thread pings the Raspberry Pi every 30 seconds and flags a disconnect before the next access event.
 
---
 
## Certificate Setup
 
Each node requires its own AWS IoT Core X.509 certificate. Place the following files in the `certs/` directory on each PYNQ:
 
```
AmazonRootCA1.pem
nodeX-certificate.pem.crt
nodeX-private.pem.key
```
 
Do not commit certificates to this repository.
 
---
 
## Notes
 
- The PYNQ boards run Python 3 via Jupyter. Scripts can also be run directly from the terminal over SSH.
- The Raspberry Pi serial port in `Pi_Controller.py` defaults to `/dev/ttyAMA0`. Change this to `/dev/ttyUSB0` or `/dev/ttyACM0` if needed depending on your connection type.
- dlib cannot be compiled from source on a t3.micro due to RAM constraints. Install via a pre-built binary wheel.
