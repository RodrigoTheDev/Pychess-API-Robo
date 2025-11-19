import serial
import time
import threading

PORT_SERVO = "COM8"
BAUDRATE_SERVO = 9600

# Lock para evitar conflitos entre threads
servo_lock = threading.Lock()

try:
    servo_serial = serial.Serial(PORT_SERVO, BAUDRATE_SERVO, timeout=1)
    time.sleep(2)
    print(f"🔌 Servo conectado com sucesso em {PORT_SERVO}.")
except serial.SerialException as e:
    print(f"❌ Erro ao conectar servo em {PORT_SERVO}: {e}")
    servo_serial = None


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

while True:
    comando = input("Digite o comando (abrir ou fechar)")
    send_servo_command(comando)