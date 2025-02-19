import serial

class SerialWrapper:
    def __init__(self, port="COM4", baud=9600):
        try:
            self.port = serial.Serial(port, baud)
        except Exception as e:
            print(e)
            self.port = None

    def __call__(self, number: int):
        if self.port is None:
            print("Dummy Serial:", number)
        else:
            print(number)
            self.port.write(number.to_bytes(1, byteorder="big"))
            self.port.flush()


if __name__ == "__main__":
    import time

    s = SerialWrapper()

    while True:
        s(180)
        time.sleep(0.5)
        s(0)
        time.sleep(0.5)