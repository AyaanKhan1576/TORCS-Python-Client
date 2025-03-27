import msgParser
import carState
import carControl
import keyboard
import csv

class Driver(object):
    def __init__(self, stage):
        self.stage = stage
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        self.gear = 1
        self.prev_rpm = None
        self.accel = 0.0
        self.brake = 0.0
        self.steer = 0.0

        # Removed manual gear override; using automatic gear shifting.
        self.last_gear_up = False
        self.last_gear_down = False

        self.was_reversing = False

        # Updated CSV header to include all telemetry fields and arrow key states.
        self.logfile = open("driving_data.csv", "w", newline="")
        self.logger = csv.writer(self.logfile)
        self.logger.writerow([
            "speedX", "speedY", "speedZ", "angle", "curLapTime", "damage",
            "distFromStart", "distRaced", "focus", "fuel", "gear", "lastLapTime",
            "opponents", "racePos", "rpm", "track", "trackPos", "wheelSpinVel", "z",
            "steer_input", "accel_input", "brake_input",
            "left_key", "right_key", "up_key", "down_key"
        ])

    def init(self):
        angles = [0 for x in range(19)]
        for i in range(5):
            angles[i] = -90 + i * 15
            angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            angles[i] = -20 + (i - 5) * 5
            angles[18 - i] = 20 - (i - 5) * 5
        return self.parser.stringify({'init': angles})

    def drive(self, msg):
        self.state.setFromMsg(msg)

        rpm = self.state.getRpm()
        speedX = self.state.getSpeedX()

        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0

        # Determine arrow key states as boolean values (1 for pressed, 0 for not pressed)
        left_key = 1 if keyboard.is_pressed('left') else 0
        right_key = 1 if keyboard.is_pressed('right') else 0
        up_key = 1 if keyboard.is_pressed('up') else 0
        down_key = 1 if keyboard.is_pressed('down') else 0

        # --- Steering ---
        if left_key:
            self.steer = 1.0
        elif right_key:
            self.steer = -1.0

        # --- Reverse / Forward Management ---
        reversing = False
        if down_key and speedX < 1:
            self.accel = 0.5
            self.gear = -1
            reversing = True
        elif up_key:
            self.accel = 1.0
            if self.gear < 1:
                self.gear = 1  # switch to forward gear after reversing
            reversing = False
        elif down_key:
            self.brake = 1.0

        # --- Automatic Gear Logic ---
        # Use current RPM and previous RPM to decide whether to upshift or downshift.
        if rpm is not None:
            # Set up flag based on RPM increase compared to previous cycle.
            if self.prev_rpm is None:
                up = True
            else:
                up = (rpm - self.prev_rpm) > 0

            # Automatically shift gear based on RPM thresholds.
            # (You can adjust the thresholds and gear limits as needed.)
            if up and rpm > 7000 and self.gear < 6:
                self.gear += 1
            elif not up and rpm < 3000 and self.gear > 1:
                self.gear -= 1

            self.prev_rpm = rpm

        # --- Apply Control ---
        self.control.setGear(self.gear)
        self.control.setAccel(self.accel)
        self.control.setBrake(self.brake)
        self.control.setSteer(self.steer)

        # --- Logging ---
        try:
            self.logger.writerow([
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
                self.state.getRpm(),
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
