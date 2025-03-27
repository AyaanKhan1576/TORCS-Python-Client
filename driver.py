
# import msgParser
# import carState
# import carControl
# import keyboard
# import csv

# class Driver(object):
#     def __init__(self, stage):
#         self.stage = stage
#         self.parser = msgParser.MsgParser()
#         self.state = carState.CarState()
#         self.control = carControl.CarControl()

#         self.gear = 1
#         self.accel = 0.0
#         self.brake = 0.0
#         self.steer = 0.0

#         self.auto_gear = True  # enable auto gear by default
#         self.last_gear_up = False
#         self.last_gear_down = False

#         self.logfile = open("driving_data.csv", "w", newline="")
#         self.logger = csv.writer(self.logfile)
#         self.logger.writerow([
#             "speedX", "speedY", "speedZ", "angle", "trackPos", "rpm",
#             "gear", "steer_input", "accel_input", "brake_input"
#         ])

#     def init(self):
#         angles = [0] * 19
#         for i in range(5):
#             angles[i] = -90 + i * 15
#             angles[18 - i] = 90 - i * 15
#         for i in range(5, 9):
#             angles[i] = -20 + (i - 5) * 5
#             angles[18 - i] = 20 - (i - 5) * 5
#         return self.parser.stringify({'init': angles})

#     def drive(self, msg):
#         self.state.setFromMsg(msg)

#         rpm = self.state.getRpm()
#         speedX = self.state.getSpeedX()

#         self.steer = 0.0
#         self.accel = 0.0
#         self.brake = 0.0

#         # --- Steering ---
#         if keyboard.is_pressed('left'):
#             self.steer = 1.0
#         elif keyboard.is_pressed('right'):
#             self.steer = -1.0

#         # --- Acceleration ---
#         if keyboard.is_pressed('up'):
#             self.accel = 1.0
#             self.gear = 1

#         # --- Reverse/Brake Logic ---
#         if keyboard.is_pressed('down'):
#             if speedX > 1:
#                 self.brake = 1.0
#             else:
#                 self.accel = 0.5
#                 self.gear = -1  # Reverse gear

#         # --- Manual Gear Control ---
#         manual_override = False
#         if keyboard.is_pressed('a'):
#             if not self.last_gear_up:
#                 self.gear = min(self.gear + 1, 6)
#                 self.last_gear_up = True
#                 manual_override = True
#         else:
#             self.last_gear_up = False

#         if keyboard.is_pressed('z'):
#             if not self.last_gear_down:
#                 self.gear = max(self.gear - 1, 1)
#                 self.last_gear_down = True
#                 manual_override = True
#         else:
#             self.last_gear_down = False

#         # --- Automatic Gear Shifting ---
#         if not manual_override and self.gear > 0:
#             if rpm is not None:
#                 if rpm > 7000 and self.gear < 6:
#                     self.gear += 1
#                 elif rpm < 3000 and self.gear > 1:
#                     self.gear -= 1

#         # --- Apply Controls ---
#         self.control.setGear(self.gear)
#         self.control.setAccel(self.accel)
#         self.control.setBrake(self.brake)
#         self.control.setSteer(self.steer)

#         # --- Logging ---
#         try:
#             self.logger.writerow([
#                 self.state.getSpeedX(),
#                 self.state.getSpeedY(),
#                 self.state.getSpeedZ(),
#                 self.state.getAngle(),
#                 self.state.getTrackPos(),
#                 self.state.getRpm(),
#                 self.gear,
#                 self.steer,
#                 self.accel,
#                 self.brake
#             ])
#         except:
#             pass

#         print(f"[SEND] steer={self.steer:.2f} accel={self.accel:.2f} brake={self.brake:.2f} gear={self.gear}")
#         return self.control.toMsg()

#     def onShutDown(self):
#         self.logfile.close()

#     def onRestart(self):
#         pass

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

        self.last_gear_up = False
        self.last_gear_down = False

        self.was_reversing = False

        self.logfile = open("driving_data.csv", "w", newline="")
        self.logger = csv.writer(self.logfile)
        self.logger.writerow([
            "speedX", "speedY", "speedZ", "angle", "trackPos", "rpm",
            "gear", "steer_input", "accel_input", "brake_input"
        ])

    def init(self):
        angles = [0] * 19
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
        gear = self.gear

        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0

        # --- Steering ---
        if keyboard.is_pressed('left'):
            self.steer = 1.0
        elif keyboard.is_pressed('right'):
            self.steer = -1.0

        # --- Reverse / Forward Management ---
        reversing = False
        forward_pressed = keyboard.is_pressed('up')
        reverse_pressed = keyboard.is_pressed('down')

        if reverse_pressed and speedX < 1:
            self.accel = 0.5
            self.gear = -1
            reversing = True
        elif forward_pressed:
            self.accel = 1.0
            if self.gear < 1:
                self.gear = 1  # switch to forward gear after reversing
            reversing = False
        elif reverse_pressed:
            self.brake = 1.0

        # --- Manual Gear Override ---
        manual_override = False
        if keyboard.is_pressed('a'):
            if not self.last_gear_up:
                self.gear = min(self.gear + 1, 6)
                self.last_gear_up = True
                manual_override = True
        else:
            self.last_gear_up = False

        if keyboard.is_pressed('z'):
            if not self.last_gear_down:
                self.gear = max(self.gear - 1, 1)
                self.last_gear_down = True
                manual_override = True
        else:
            self.last_gear_down = False

        # --- Automatic Gear Logic (if not manual and not reversing) ---
        if not manual_override and self.gear > 0 and not reversing:
            up = True
            #gear()
            if self.prev_rpm is not None and rpm is not None:
                up = (rpm - self.prev_rpm) > 0
            if rpm is not None:
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
                self.state.getTrackPos(),
                self.state.getRpm(),
                self.gear,
                self.steer,
                self.accel,
                self.brake
            ])
        except:
            pass

        print(f"[SEND] steer={self.steer:.2f} accel={self.accel:.2f} brake={self.brake:.2f} gear={self.gear}")
        return self.control.toMsg()

    # def gear(self):
    #     rpm = self.state.getRpm()
    #     gear = self.state.getGear()
        
    #     if self.prev_rpm == None:
    #         up = True
    #     else:
    #         if (self.prev_rpm - rpm) < 0:
    #             up = True
    #         else:
    #             up = False
        
    #     if up and rpm > 7000:
    #         gear += 1
        
    #     if not up and rpm < 3000:
    #         gear -= 1
        
    #     self.control.setGear(gear)

    def onShutDown(self):
        self.logfile.close()

    def onRestart(self):
        pass
