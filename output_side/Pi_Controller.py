import serial
import RPi.GPIO as GPIO
import time

SERIAL_PORT = "/dev/ttyAMA0"
SERIAL_BAUD = 9600

LED_RED   = 17
LED_GREEN = 27
LED_BLUE  = 22
SERVO_PIN = 18

SERVO_OPEN_DUTY  = 7.5
SERVO_CLOSE_DUTY = 2.5

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(LED_RED,   GPIO.OUT)
GPIO.setup(LED_GREEN, GPIO.OUT)
GPIO.setup(LED_BLUE,  GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)


def all_leds_off():
    GPIO.output(LED_RED,   GPIO.LOW)
    GPIO.output(LED_GREEN, GPIO.LOW)
    GPIO.output(LED_BLUE,  GPIO.LOW)


def set_led(state: str):
    all_leds_off()
    if state == "GREEN":
        GPIO.output(LED_GREEN, GPIO.HIGH)
    elif state == "RED":
        GPIO.output(LED_RED, GPIO.HIGH)
    elif state == "AMBER":
        GPIO.output(LED_RED,   GPIO.HIGH)
        GPIO.output(LED_GREEN, GPIO.HIGH)
    elif state == "RED_FLASH":
        for _ in range(6):
            GPIO.output(LED_RED, GPIO.HIGH)
            time.sleep(0.25)
            GPIO.output(LED_RED, GPIO.LOW)
            time.sleep(0.25)
    elif state == "OFF":
        all_leds_off()


def set_servo(action: str):
    if action == "OPEN":
        servo.ChangeDutyCycle(SERVO_OPEN_DUTY)
        time.sleep(0.5)
        servo.ChangeDutyCycle(0)
    elif action == "CLOSE":
        servo.ChangeDutyCycle(SERVO_CLOSE_DUTY)
        time.sleep(0.5)
        servo.ChangeDutyCycle(0)


def handle_command(command: str):
    command = command.strip()
    if command == "PING":
        return "ACK"
    elif command.startswith("LED:"):
        state = command.split(":", 1)[1]
        set_led(state)
    elif command.startswith("SERVO:"):
        action = command.split(":", 1)[1]
        set_servo(action)
    return None


try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    print(f"[+] Pi controller listening on {SERIAL_PORT}")

    while True:
        if ser.in_waiting > 0:
            raw = ser.readline().decode("utf-8").strip()
            if raw:
                response = handle_command(raw)
                if response:
                    ser.write((response + "\n").encode("utf-8"))

except KeyboardInterrupt:
    print("\n[i] Interrupted by user")

finally:
    servo.stop()
    GPIO.cleanup()
    print("[+] Pi controller stopped.")
