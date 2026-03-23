import time
import json
import serial
import serial.tools.list_ports
import paho.mqtt.client as mqtt
from pathlib import Path
import traceback
import threading
import logging

MQTT_ENDPOINT = "acse20kw9idxo-ats.iot.eu-west-2.amazonaws.com"
MQTT_PORT     = 8883
NODE_ID       = "node2"

BASE_DIR    = Path("/home/xilinx/jupyter_notebooks/IP PRJ")
CERTS_DIR   = BASE_DIR / "certs"
CA_CERT     = CERTS_DIR / "AmazonRootCA1.pem"
CLIENT_CERT = CERTS_DIR / "node2-certificate.pem.crt"
CLIENT_KEY  = CERTS_DIR / "node2-private.pem.key"

TOPIC_DECISION = "attendance/node1/decision"
TOPIC_COMMAND  = "attendance/node2/command"

COOLDOWN_SECONDS       = 30
DENIAL_STREAK_LIMIT    = 3
ROLLING_WINDOW_LIMIT   = 5
ROLLING_WINDOW_SECONDS = 120

SERIAL_BAUD        = 9600
SERIAL_TIMEOUT     = 2
HEARTBEAT_INTERVAL = 30.0

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / "output_node.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

cooldown_map      = {}
denial_streak     = 0
denial_timestamps = []
led_only_mode     = False
serial_conn       = None


def find_serial_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "USB" in port.description or "ttyUSB" in port.device or "ttyACM" in port.device:
            return port.device
    return None


def init_serial():
    global serial_conn, led_only_mode
    port = find_serial_port()
    if port is None:
        print("[!] Raspberry Pi not detected — entering LED-only mode")
        logging.warning("Raspberry Pi not detected at startup — LED-only mode active")
        led_only_mode = True
        return
    try:
        serial_conn = serial.Serial(port, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
        time.sleep(2)
        print(f"[+] Raspberry Pi connected on {port}")
        logging.info(f"Raspberry Pi connected on {port}")
        led_only_mode = False
    except serial.SerialException as e:
        print(f"[!] Could not open serial port {port}: {e} — entering LED-only mode")
        logging.warning(f"Serial open failed: {e} — LED-only mode active")
        led_only_mode = True


def send_serial_command(command: str):
    global serial_conn, led_only_mode
    if led_only_mode or serial_conn is None:
        return
    try:
        serial_conn.write((command + "\n").encode("utf-8"))
        logging.info(f"Serial command sent: {command}")
    except serial.SerialException as e:
        print(f"[!] Serial write failed: {e} — falling back to LED-only mode")
        logging.warning(f"Serial write failed: {e} — switching to LED-only mode")
        led_only_mode = True


def heartbeat_monitor():
    global serial_conn, led_only_mode
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        if led_only_mode or serial_conn is None:
            continue
        try:
            serial_conn.write(b"PING\n")
            serial_conn.timeout = SERIAL_TIMEOUT
            response = serial_conn.readline().decode("utf-8").strip()
            if response == "ACK":
                logging.info("Heartbeat: Raspberry Pi ACK received")
            else:
                print("[!] Heartbeat: No ACK from Raspberry Pi — entering LED-only mode")
                logging.warning("Heartbeat: No ACK — switching to LED-only mode")
                led_only_mode = True
        except Exception as e:
            print(f"[!] Heartbeat check failed: {e} — entering LED-only mode")
            logging.warning(f"Heartbeat exception: {e} — switching to LED-only mode")
            led_only_mode = True


def set_led(state: str):
    send_serial_command(f"LED:{state}")


def trigger_servo(action: str):
    if not led_only_mode:
        send_serial_command(f"SERVO:{action}")


def dispatch_state(state: str, person_id: str = "Unknown"):
    print(f"\n{'='*40}")
    print(f"  STATE:   {state}")
    print(f"  PERSON:  {person_id}")
    print(f"  MODE:    {'LED-only' if led_only_mode else 'Full'}")
    print(f"{'='*40}\n")

    logging.info(f"State dispatched: {state} | Person: {person_id} | LED-only: {led_only_mode}")

    if state == "GRANT":
        set_led("GREEN")
        trigger_servo("OPEN")
        time.sleep(3)
        trigger_servo("CLOSE")
        set_led("OFF")

    elif state == "DENY":
        set_led("RED")
        time.sleep(3)
        set_led("OFF")

    elif state == "PENDING":
        set_led("AMBER")

    elif state == "ALARM":
        set_led("RED_FLASH")
        time.sleep(3)
        set_led("OFF")


def check_cooldown(person_id: str) -> bool:
    if person_id in cooldown_map:
        elapsed = time.time() - cooldown_map[person_id]
        if elapsed < COOLDOWN_SECONDS:
            print(f"[i] COOLDOWN_ACTIVE for {person_id} ({elapsed:.1f}s elapsed)")
            logging.info(f"COOLDOWN_ACTIVE: {person_id}, {elapsed:.1f}s < {COOLDOWN_SECONDS}s")
            return True
    return False


def update_denial_tracking():
    global denial_streak, denial_timestamps
    denial_streak += 1
    now = time.time()
    denial_timestamps.append(now)
    denial_timestamps = [t for t in denial_timestamps if now - t <= ROLLING_WINDOW_SECONDS]
    if denial_streak >= DENIAL_STREAK_LIMIT:
        return True
    if len(denial_timestamps) >= ROLLING_WINDOW_LIMIT:
        return True
    return False


def reset_denial_tracking():
    global denial_streak, denial_timestamps
    denial_streak = 0
    denial_timestamps = []


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[+] Connected to AWS IoT Core")
        client.subscribe(TOPIC_DECISION, qos=1)
        client.subscribe(TOPIC_COMMAND, qos=1)
    else:
        print(f"[-] MQTT connection failed: rc={rc}")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f"[!] Unexpected MQTT disconnect (rc={rc}), will auto-reconnect...")
    else:
        print("[i] MQTT disconnected cleanly")


def on_message(client, userdata, msg):
    global cooldown_map

    try:
        data      = json.loads(msg.payload)
        decision  = data.get("decision")
        person_id = data.get("person_id", "Unknown")
        already   = data.get("already_inside", False)

        dispatch_state("PENDING", person_id)
        time.sleep(0.5)

        if decision == "GRANT":
            if already:
                print(f"[i] {person_id} already inside — no duplicate record written")
                logging.info(f"Already inside: {person_id}")
                set_led("OFF")
                return

            if check_cooldown(person_id):
                set_led("OFF")
                return

            cooldown_map[person_id] = time.time()
            reset_denial_tracking()
            dispatch_state("GRANT", person_id)

        elif decision == "DENY":
            trigger_alarm = update_denial_tracking()
            if trigger_alarm:
                dispatch_state("ALARM", person_id)
            else:
                dispatch_state("DENY", person_id)

    except Exception as e:
        print(f"[-] Error handling message: {e}")
        traceback.print_exc()


def connect_mqtt():
    client = mqtt.Client(client_id="pynq-node2", clean_session=False)
    client.tls_set(
        ca_certs=str(CA_CERT),
        certfile=str(CLIENT_CERT),
        keyfile=str(CLIENT_KEY),
    )
    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=60)

    print(f"[+] Connecting to {MQTT_ENDPOINT}...")
    client.connect(MQTT_ENDPOINT, MQTT_PORT, keepalive=60)
    client.loop_start()
    time.sleep(2)
    return client


if __name__ == "__main__":
    print("[+] Output node starting...")

    init_serial()

    hb_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
    hb_thread.start()

    mqtt_client = connect_mqtt()

    print("[+] Output node ready — waiting for decisions...\n")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[i] Interrupted by user")

    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        if serial_conn and serial_conn.is_open:
            serial_conn.close()
        print("[+] Output node stopped.")
