import time
import numpy as np
import spidev
from ai_feature_engine import AIFeaturePath

class FPGABridge:
    def __init__(self, bus=0, device=0):
        self.spi = spidev.SpiDev()
        try:
            self.spi.open(bus, device)
            self.spi.max_speed_hz = 10000000
        except Exception:
            print("[WARN] SPI Hardware not detected. Running mock bridge.")

    def send_classification(self, speech_status, noise_type, impulse_status):
        # Pack into 32-bit Control Word
        packet = [speech_status & 0xFF, noise_type & 0xFF, impulse_status & 0xFF, 0x00]
        try:
            self.spi.xfer2(packet)
        except Exception:
            pass
        return packet

if __name__ == "__main__":
    bridge = FPGABridge()
    engine = AIFeaturePath()
    print("[SPI BRIDGE] Active. Listening and transmitting to Zybo Z7-10...")
    while True:
        dummy_audio = np.random.uniform(-0.05, 0.05, 2048)
        metrics = engine.analyze_audio(dummy_audio)
        pkt = bridge.send_classification(metrics['speech_status'], metrics['noise_type'], metrics['impulsive_status'])
        time.sleep(0.02)
