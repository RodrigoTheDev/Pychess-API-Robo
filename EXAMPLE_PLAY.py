import serial
import time
import threading

PORT_SERVO = "COM8"
BAUDRATE_SERVO = 9600

PORT_MOTOR = "COM5"
BAUDRATE_MOTOR = 115200

# Lock para evitar conflitos entre threads
servo_lock = threading.Lock()

try:
    servo_serial = serial.Serial(PORT_SERVO, BAUDRATE_SERVO, timeout=1)
    time.sleep(2)
    print(f"🔌 Servo conectado com sucesso em {PORT_SERVO}.")
except serial.SerialException as e:
    print(f"❌ Erro ao conectar servo em {PORT_SERVO}: {e}")
    servo_serial = None

try:
    motor_serial = serial.Serial(PORT_MOTOR, BAUDRATE_MOTOR, timeout=1)
    time.sleep(2)
    print(f"🔌 Motor conectado com sucesso em {PORT_MOTOR}.")
except serial.SerialException as e:
    print(f"❌ Erro ao conectar motor em {PORT_MOTOR}: {e}")
    motor_serial = None


def send_servo_command(command: str):
    """Envia um comando simples ('abrir' ou 'fechar') ao servo."""
    global servo_serial
    if not servo_serial or not servo_serial.is_open:
        raise RuntimeError("Conexão serial do servo não está ativa.")

    with servo_lock:
        msg = command.strip() + "\n"
        print(f"📤 Enviando comando ao servo: {msg.strip()}")
        servo_serial.write(msg.encode("utf-8"))
        servo_serial.flush()
        time.sleep(0.5)  # pequena pausa para garantir execução no Arduino

def send_motor_command(command: str):
    global motor_serial
    if not motor_serial or not motor_serial.is_open:
        raise RuntimeError("Conexão serial do servo não está ativa.")

    with servo_lock:
        msg = command.strip() + "\n"
        print(f"📤 Enviando comando ao servo: {msg.strip()}")
        motor_serial.write(msg.encode("utf-8"))
        motor_serial.flush()
        time.sleep(0.5)  # pequena pausa para garantir execução no Arduino
    
    #time.sleep(2)

def exampleMove():
        #INSTRUÇÕES:
        send_motor_command("F Y 3000")
        send_motor_command("F X 4500")
        time.sleep(2)
        send_servo_command("open")
        send_motor_command("F E0 1000")
        send_motor_command("B Y 2400")
        send_motor_command("F E1 700")
        time.sleep(2)
        send_servo_command("close")
        send_motor_command("F Y 1000")
        send_motor_command("B E1 300")
        time.sleep(2)
        send_servo_command("open")
        # VOLTANDO
        send_motor_command("F Y 1000")
        send_servo_command("close")
        send_motor_command("B X 4700")
        send_motor_command("B E0 1000")
        send_motor_command("B E1 400")
        send_motor_command("B Y 3000")