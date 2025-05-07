# driver.py

import os
import csv
import xml.etree.ElementTree as ET
from joblib import load
import msgParser
import carState
import carControl

class Driver(object):
    def __init__(self, stage):
        self.stage = stage
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        # Load the trained multi-output RandomForest model
        model_path = os.path.join(os.path.dirname(__file__), 'torcmodel.pkl')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.policy = load(model_path)

        # Initialize gear
        self.gear = 1

        # Get track, driver, car info
        self.track_name, self.driver_module, self.car_model = self.get_track_driver_car()

        # Setup logging
        log_exists = os.path.isfile("driving_data.csv")
        self.logfile = open("driving_data.csv", "a", newline="")
        self.logger = csv.writer(self.logfile)
        if not log_exists:
            self.logger.writerow([
                "track","driver","car",
                "speedX","angle","trackPos","rpm",
                "steer_pred","accel_pred","brake_pred",
                "left_key_pred","right_key_pred","up_key_pred","down_key_pred",
                "gear"
            ])

    def get_track_driver_car(self):
        try:
            base_dir = os.path.join(r"C:\Users\Pakistan\Desktop\AI_Proj\torcs")
            qr = os.path.join(base_dir, "config/raceman/quickrace.xml")
            tree = ET.parse(qr)
            root = tree.getroot()

            tsec = root.find(".//section[@name='Tracks']/section")
            track = tsec.find("attstr[@name='name']").attrib['val'] if tsec is not None else "unknown"

            drv = root.find(".//section[@name='Drivers']")
            idx = drv.find("attnum[@name='focused idx']").attrib['val'] if drv is not None else "0"
            mod = drv.find("attstr[@name='focused module']").attrib['val'] if drv is not None else "unknown"

            ss = os.path.join(base_dir, "drivers", mod, "scr_server.xml")
            tree2 = ET.parse(ss)
            root2 = tree2.getroot()
            rob = root2.find(f".//section[@name='Robots']/section[@name='index']/section[@name='{idx}']")
            if rob is None:
                rob = root2.find(".//section[@name='Robots']/section[@name='index']/section")
            car = rob.find("attstr[@name='car name']").attrib['val'] if rob is not None else "unknown"

            return track, mod, car
        except Exception as e:
            print(f"[XML Error] {e}")
            return "unknown","unknown","unknown"

    def init(self):
        angles = [0]*19
        for i in range(5):
            angles[i] = -90 + 15*i
            angles[18-i] = 90 - 15*i
        for i in range(5,9):
            angles[i] = -20 + 5*(i-5)
            angles[18-i] = 20 - 5*(i-5)
        return self.parser.stringify({'init': angles})

    def drive(self, msg):
        # Parse sensors
        self.state.setFromMsg(msg)
        cs = self.state

        # Build feature vector matching training (4 features)
        speedX = cs.getSpeedX()
        angle = cs.getAngle()
        trackPos = cs.getTrackPos()
        rpm = cs.getRpm()
        features = [[speedX, angle, trackPos, rpm]]

        # Predict the seven control outputs
        pred = self.policy.predict(features)[0]
        steer_pred, accel_pred, brake_pred, lk_pred, rk_pred, uk_pred, dk_pred = pred

        # Gear logic from key predictions
        if dk_pred >= 0.5 and speedX < 1.0:
            gear = -1
        elif uk_pred >= 0.5:
            gear = max(1, self.gear)
        else:
            gear = min(6, max(1, int(speedX // 50) + 1))
        self.gear = gear

        # Apply control commands
        self.control.setSteer(float(steer_pred))
        self.control.setAccel(float(accel_pred))
        self.control.setBrake(float(brake_pred))
        self.control.setGear(gear)

        # Log inputs and predictions
        try:
            self.logger.writerow([
                self.track_name, self.driver_module, self.car_model,
                speedX, angle, trackPos, rpm,
                steer_pred, accel_pred, brake_pred,
                lk_pred, rk_pred, uk_pred, dk_pred,
                gear
            ])
        except Exception as e:
            print(f"[Logging Error] {e}")

        return self.control.toMsg()

    def onShutDown(self):
        self.logfile.close()

    def onRestart(self):
        pass
