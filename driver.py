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

        # Start in forward gear (1)
        self.gear = 1
        self.accel = 0.0
        self.brake = 0.0
        self.steer = 0.0

        self.was_reversing = False

        # Add a steering sensitivity scale. Adjust this value (0.0 to 1.0) to change how sensitive the steering is.
        self.steering_scale = 1.0

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

        speedX = self.state.getSpeedX()
        rpm = self.state.getRpm()

        # Reset control values
        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0

        # Determine arrow key states (1 if pressed, else 0)
        left_key = 1 if keyboard.is_pressed('left') else 0
        right_key = 1 if keyboard.is_pressed('right') else 0
        up_key = 1 if keyboard.is_pressed('up') else 0
        down_key = 1 if keyboard.is_pressed('down') else 0

        # --- Steering with Smoothing ---
        # Compute target steering value based on key input.
        target_steer = 0.0
        if left_key:
            target_steer = self.steering_scale
        elif right_key:
            target_steer = -self.steering_scale
        else:
            target_steer = 0.0

        # Apply smoothing: move 20% of the way toward target each cycle.
        smoothing_factor = 0.2
        self.steer = (1 - smoothing_factor) * self.steer + smoothing_factor * target_steer

        # --- Reverse / Forward Management ---
        if down_key and speedX < 1:
            self.accel = 0.5
            self.gear = -1
        elif up_key:
            self.accel = 1.0
            if self.gear < 1:
                self.gear = 1
        elif down_key:
            self.brake = 1.0

        # --- Automatic Gear Logic Based on Speed ---
        if self.gear != -1:
            threshold = 50  # Adjust threshold step as needed
            new_gear = int(speedX // threshold) + 1
            if new_gear < 1:
                new_gear = 1
            elif new_gear > 6:
                new_gear = 6
            self.gear = new_gear

        # --- Steering Damping Based on Track Position ---
        # Get the current lateral position of the car.
        track_pos = self.state.getTrackPos()
        if track_pos is not None:
            # If the car is off-track (|trackPos| > 1), then reduce steering to help straighten out.
            if abs(track_pos) > 1.0:
                # Dampen steering command (you can adjust the damping factor).
                damping_factor = 0.5
                self.steer *= damping_factor
                # Optionally, reduce acceleration to avoid excessive turning at high speeds.
                self.accel = min(self.accel, 0.5)

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
