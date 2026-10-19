"""
ASAFI - Autonomous Spacecraft Avionics Fire Interceptor
Hardware Layer: PCA9685 Gimbal & MLX90614 Non-Contact IR Sensor
With automatic bus detection and graceful MOCK fallbacks.
"""

import os
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ASAFI.Hardware")

try:
    import smbus2
    HAS_SMBUS2 = True
except ImportError:
    HAS_SMBUS2 = False
    logger.warning("[HARDWARE] smbus2 module not installed. Hardware I2C disabled.")


class PCA9685Driver:
    """Lightweight PCA9685 PWM Driver using smbus2 directly."""
    PCA9685_MODE1 = 0x00
    PCA9685_PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, bus_num=1, address=0x40):
        self.bus_num = bus_num
        self.address = address
        self.bus = smbus2.SMBus(bus_num)
        self.init_device()

    def init_device(self):
        # Reset mode
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, 0x00)
        time.sleep(0.005)
        # Set 50Hz for standard analog/digital servos (prescale ~ 121)
        # prescale = round(25MHz / (4096 * 50Hz)) - 1 = 121 (0x79)
        old_mode = self.bus.read_byte_data(self.address, self.PCA9685_MODE1)
        sleep_mode = (old_mode & 0x7F) | 0x10  # sleep
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, sleep_mode)
        self.bus.write_byte_data(self.address, self.PCA9685_PRESCALE, 121)
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, old_mode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, old_mode | 0xA1)  # auto-increment

    def set_pwm(self, channel, on_tick, off_tick):
        reg = self.LED0_ON_L + 4 * channel
        data = [on_tick & 0xFF, (on_tick >> 8) & 0xFF, off_tick & 0xFF, (off_tick >> 8) & 0xFF]
        self.bus.write_i2c_block_data(self.address, reg, data)

    def set_servo_angle(self, channel, angle_deg, min_us=500, max_us=2500, angle_range=180.0):
        """Map angle (0 to 180) to 50Hz PWM pulse width ticks (0-4095)."""
        clamped_angle = max(0.0, min(angle_range, float(angle_deg)))
        pulse_us = min_us + (clamped_angle / angle_range) * (max_us - min_us)
        # In 50Hz, period = 20000us. Ticks per us = 4096 / 20000 = 0.2048
        off_tick = int(round(pulse_us * 4096.0 / 20000.0))
        self.set_pwm(channel, 0, off_tick)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass


class GimbalController:
    """
    2-Axis Pan-Tilt Gimbal Controller with PCA9685 I2C driver and Mock fallback.
    Pan Axis: Channel 0 (-90 deg to +90 deg, mapped to 0-180 servo angle, 90 is center)
    Tilt Axis: Channel 1 (-45 deg to +45 deg, mapped to 45-135 servo angle, 90 is center)
    """
    def __init__(self, bus_num=1, address=0x40, force_mock=False):
        self.bus_num = bus_num
        self.address = address
        self.is_mock = force_mock
        self.driver = None

        # Tracking state
        self.pan = 0.0      # -90.0 (left) to +90.0 (right)
        self.tilt = 0.0     # -45.0 (down) to +45.0 (up)
        self.target_pan = 0.0
        self.target_tilt = 0.0
        self.kp = 0.04      # Proportional tracking gain

        self._init_hardware()

    def _init_hardware(self):
        if self.is_mock:
            logger.info("[GIMBAL] MOCK mode enabled by flag.")
            return

        if not HAS_SMBUS2 or not os.path.exists(f"/dev/i2c-{self.bus_num}"):
            logger.warning(f"[GIMBAL] I2C bus /dev/i2c-{self.bus_num} not accessible. Falling back to MOCK mode.")
            self.is_mock = True
            return

        try:
            self.driver = PCA9685Driver(bus_num=self.bus_num, address=self.address)
            # Center both servos (0.0 offset -> 90 degrees)
            self._write_servos(0.0, 0.0)
            logger.info(f"[GIMBAL] PCA9685 successfully initialized on /dev/i2c-{self.bus_num} @ 0x{self.address:02X}")
        except Exception as e:
            logger.warning(f"[GIMBAL] PCA9685 not detected on /dev/i2c-{self.bus_num} (0x{self.address:02X}): {e}. Falling back to MOCK mode.")
            self.is_mock = True
            self.driver = None

    def _write_servos(self, pan_deg, tilt_deg):
        if self.is_mock or not self.driver:
            return
        try:
            # Map pan (-90 to +90) -> (0 to 180 servo)
            pan_servo = pan_deg + 90.0
            # Map tilt (-45 to +45) -> (45 to 135 servo)
            tilt_servo = tilt_deg + 90.0
            self.driver.set_servo_angle(0, pan_servo)
            self.driver.set_servo_angle(1, tilt_servo)
        except Exception as e:
            logger.error(f"[GIMBAL] I2C communication error while writing servos: {e}. Switching to MOCK.")
            self.is_mock = True

    def update_tracking(self, cx, cy, frame_w=640, frame_h=480):
        """
        Calculates tracking offset error from frame center and updates gimbal angles.
        Target Center Error:
            dx = cx - (frame_w / 2)   (+ is right, - is left)
            dy = cy - (frame_h / 2)   (+ is down, - is up)
        """
        center_x = frame_w / 2.0
        center_y = frame_h / 2.0

        dx = cx - center_x
        dy = cy - center_y

        # Proportional tracking step
        self.pan += self.kp * dx
        self.tilt -= self.kp * dy  # Inverted so moving up tilts camera up

        # Clamp to safe structural limits
        self.pan = max(-90.0, min(90.0, self.pan))
        self.tilt = max(-45.0, min(45.0, self.tilt))

        self.target_pan = self.pan
        self.target_tilt = self.tilt

        self._write_servos(self.pan, self.tilt)

        return {
            "pan": round(self.pan, 2),
            "tilt": round(self.tilt, 2),
            "dx": round(dx, 1),
            "dy": round(dy, 1),
            "hardware": "PHYSICAL" if not self.is_mock else "MOCK"
        }

    def center(self):
        """Reset gimbal to 0,0 center position."""
        self.pan = 0.0
        self.tilt = 0.0
        self.target_pan = 0.0
        self.target_tilt = 0.0
        self._write_servos(0.0, 0.0)

    def get_status(self):
        return {
            "pan": round(self.pan, 2),
            "tilt": round(self.tilt, 2),
            "hardware": "PHYSICAL" if not self.is_mock else "MOCK",
            "is_mock": self.is_mock
        }


class ThermalSensor:
    """
    MLX90614 Non-Contact Infrared Pyrometer with fallback to Mock readings.
    Addresses: 0x5A default.
    Registers: 0x06 (Ambient temp), 0x07 (Object 1 temp).
    Formula: Temp_C = (raw * 0.02) - 273.15
    """
    MLX90614_TA = 0x06
    MLX90614_TOBJ1 = 0x07

    def __init__(self, bus_num=3, address=0x5A, force_mock=False):
        self.bus_num = bus_num
        self.address = address
        self.is_mock = force_mock
        self.bus = None

        # State
        self.ambient_temp = 24.2
        self.object_temp = 24.8
        self.threshold_c = 55.0

        self._init_hardware()

    def _init_hardware(self):
        if self.is_mock:
            logger.info("[THERMAL] MOCK mode enabled by flag.")
            return

        if not HAS_SMBUS2:
            logger.warning("[THERMAL] smbus2 unavailable. Falling back to MOCK mode.")
            self.is_mock = True
            return

        # Check bus availability (try designated bus, fallback to bus 1)
        candidate_buses = [self.bus_num, 1] if self.bus_num != 1 else [1]
        connected = False

        for b in candidate_buses:
            dev_path = f"/dev/i2c-{b}"
            if os.path.exists(dev_path):
                try:
                    bus = smbus2.SMBus(b)
                    # Attempt reading ambient temp register
                    raw = bus.read_word_data(self.address, self.MLX90614_TA)
                    temp = (raw * 0.02) - 273.15
                    if -40.0 <= temp <= 125.0:
                        self.bus = bus
                        self.bus_num = b
                        connected = True
                        logger.info(f"[THERMAL] MLX90614 connected on {dev_path} @ 0x{self.address:02X} (Ambient: {temp:.1f}C)")
                        break
                except Exception:
                    pass

        if not connected:
            logger.warning(f"[THERMAL] MLX90614 not detected at 0x{self.address:02X} on /dev/i2c-{self.bus_num} or /dev/i2c-1. Falling back to MOCK mode.")
            self.is_mock = True

    def read(self, fire_detected=False):
        """Returns ambient and object temperatures in Celsius."""
        if not self.is_mock and self.bus:
            try:
                raw_ta = self.bus.read_word_data(self.address, self.MLX90614_TA)
                raw_to = self.bus.read_word_data(self.address, self.MLX90614_TOBJ1)
                self.ambient_temp = round((raw_ta * 0.02) - 273.15, 1)
                self.object_temp = round((raw_to * 0.02) - 273.15, 1)
                return {
                    "ambient_c": self.ambient_temp,
                    "object_c": self.object_temp,
                    "threshold_c": self.threshold_c,
                    "fire_confirmed": self.object_temp >= self.threshold_c,
                    "hardware": "PHYSICAL"
                }
            except Exception as e:
                logger.error(f"[THERMAL] Error reading MLX90614: {e}. Switching to MOCK.")
                self.is_mock = True

        # Mock Mode Simulation
        if fire_detected:
            # When fire is detected optically, mock realistic hot thermal target
            target_obj = 68.5
            self.object_temp += (target_obj - self.object_temp) * 0.3
        else:
            # Decay back to ambient
            self.object_temp += (24.8 - self.object_temp) * 0.1

        return {
            "ambient_c": round(self.ambient_temp, 1),
            "object_c": round(self.object_temp, 1),
            "threshold_c": self.threshold_c,
            "fire_confirmed": self.object_temp >= self.threshold_c,
            "hardware": "MOCK"
        }


class HardwareManager:
    """Unified Hardware Coordinator."""
    def __init__(self, force_mock=False):
        self.gimbal = GimbalController(bus_num=1, address=0x40, force_mock=force_mock)
        self.thermal = ThermalSensor(bus_num=3, address=0x5A, force_mock=force_mock)

    def update(self, fire_detected, cx=None, cy=None, frame_w=640, frame_h=480):
        gimbal_data = {}
        if fire_detected and cx is not None and cy is not None:
            gimbal_data = self.gimbal.update_tracking(cx, cy, frame_w, frame_h)
        else:
            gimbal_data = self.gimbal.get_status()

        thermal_data = self.thermal.read(fire_detected=fire_detected)

        return {
            "gimbal": gimbal_data,
            "thermal": thermal_data,
            "hardware_mode": {
                "gimbal": "PHYSICAL" if not self.gimbal.is_mock else "MOCK",
                "thermal": "PHYSICAL" if not self.thermal.is_mock else "MOCK"
            }
        }
