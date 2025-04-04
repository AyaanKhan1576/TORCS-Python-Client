import msgParser
import carState
import carControl
import keyboard
import csv
import os
import xml.etree.ElementTree as ET

class Driver(object):
    def __init__(self, stage):
        self.stage = stage
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        self.gear = 1
        self.accel = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.was_reversing = False
        self.steering_scale = 1.0

        # === Get car and track info from config ===
        self.track_name, self.driver_module, self.car_model = self.get_track_driver_car()

        # === Logging Setup ===
        file_exists = os.path.isfile("driving_data.csv")
        self.logfile = open("driving_data.csv", "a", newline="")
        self.logger = csv.writer(self.logfile)

        if not file_exists:
            self.logger.writerow([
                "track", "driver", "car",
                "speedX", "speedY", "speedZ", "angle", "curLapTime", "damage",
                "distFromStart", "distRaced", "focus", "fuel", "gear", "lastLapTime",
                "opponents", "racePos", "rpm", "trackSensor", "trackPos", "wheelSpinVel", "z",
                "steer_input", "accel_input", "brake_input",
                "left_key", "right_key", "up_key", "down_key"
            ])

    def get_track_driver_car(self):
        try:
            base_dir = os.path.join(
                r"C:\Users\ayaan\Documents\University\Semester 6\Aritificial Intelligence\Project\torcs"
            )
            quickrace_xml = os.path.join(base_dir, "config/raceman/quickrace.xml")
            # Parse quickrace.xml to get track name and focused driver info.
            tree = ET.parse(quickrace_xml)
            root = tree.getroot()

            track_section = root.find(".//section[@name='Tracks']/section")
            track_name = (track_section.find("attstr[@name='name']").attrib['val']
                          if track_section is not None else "unknown_track")

            drivers_section = root.find(".//section[@name='Drivers']")
            focused_idx = drivers_section.find("attnum[@name='focused idx']").attrib['val'] if drivers_section.find("attnum[@name='focused idx']") is not None else "0"
            focused_module = drivers_section.find("attstr[@name='focused module']").attrib['val'] if drivers_section.find("attstr[@name='focused module']") is not None else "unknown_driver"

            # === Now parse scr_server.xml to fetch the car name ===
            # Assuming scr_server.xml is located at: C:\Program Files (x86)\torcs\drivers\scr_server\scr_server.xml
            scr_server_xml = os.path.join(base_dir, "drivers", focused_module, "scr_server.xml")
            tree2 = ET.parse(scr_server_xml)
            root2 = tree2.getroot()

            # In scr_server.xml, find the focused driver's robot definition using the focused index.
            # The structure is: <params name="scr_server" ...>\n  <section name="Robots">\n    <section name="index">\n      <section name="0"> ... </section>\n      ...\n    </section>\n  </section>\n</params>
            driver_section = root2.find(f".//section[@name='Robots']/section[@name='index']/section[@name='{focused_idx}']")
            if driver_section is None:
                # If not found by focused index, fall back to the first section.
                driver_section = root2.find(".//section[@name='Robots']/section[@name='index']/section")
            if driver_section is not None:
                car_name = driver_section.find("attstr[@name='car name']").attrib['val']
            else:
                car_name = "unknown_car"

            return track_name, focused_module, car_name
        except Exception as e:
            print(f"[XML Error] {e}")
            return "unknown_track", "unknown_driver", "unknown_car"

    def init(self):
        angles = [0 for _ in range(19)]
        for i in range(5):
            angles[i] = -90 + i * 15
            angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            angles[i] = -20 + (i - 5) * 5
            angles[18 - i] = 20 - (i - 5) * 5
        return self.parser.stringify({'init': angles})

    def drive(self, msg):
        self.state.setFromMsg(msg)

        speedX = self.state.getSpeedX()
        rpm = self.state.getRpm()

        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0

        left_key = 1 if keyboard.is_pressed('left') else 0
        right_key = 1 if keyboard.is_pressed('right') else 0
        up_key = 1 if keyboard.is_pressed('up') else 0
        down_key = 1 if keyboard.is_pressed('down') else 0

        target_steer = self.steering_scale if left_key else (-self.steering_scale if right_key else 0.0)
        self.steer = 0.8 * self.steer + 0.2 * target_steer

        if down_key and speedX < 1:
            self.accel = 0.5
            self.gear = -1
        elif up_key:
            self.accel = 1.0
            if self.gear < 1:
                self.gear = 1
        elif down_key:
            self.brake = 1.0

        if self.gear != -1:
            new_gear = min(6, max(1, int(speedX // 50) + 1))
            self.gear = new_gear

        track_pos = self.state.getTrackPos()
        if track_pos is not None and abs(track_pos) > 1.0:
            self.steer *= 0.5
            self.accel = min(self.accel, 0.5)

        self.control.setGear(self.gear)
        self.control.setAccel(self.accel)
        self.control.setBrake(self.brake)
        self.control.setSteer(self.steer)

        try:
            self.logger.writerow([
                self.track_name,
                self.driver_module,
                self.car_model,
                self.state.getSpeedX(),
                self.state.getSpeedY(),
                self.state.getSpeedZ(),
                self.state.getAngle(),
                self.state.getCurLapTime(),
                self.state.getDamage(),
                self.state.getDistFromStart(),
                self.state.getDistRaced(),
                str(self.state.getFocus()),
                self.state.getFuel(),
                self.state.getGear(),
                self.state.getLastLapTime(),
                str(self.state.getOpponents()),
                self.state.getRacePos(),
                rpm,
                str(self.state.getTrack()),
                self.state.getTrackPos(),
                str(self.state.getWheelSpinVel()),
                self.state.getZ(),
                self.steer,
                self.accel,
                self.brake,
                left_key,
                right_key,
                up_key,
                down_key
            ])
        except Exception as e:
            print(f"[Logging Error] {e}")

        print(f"[SEND] steer={self.steer:.2f} accel={self.accel:.2f} brake={self.brake:.2f} gear={self.gear}")
        return self.control.toMsg()

    def onShutDown(self):
        self.logfile.close()

    def onRestart(self):
        pass
