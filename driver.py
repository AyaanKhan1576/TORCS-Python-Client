import msgParser
import carState
import carControl
import keyboard
import csv, os, time
import xml.etree.ElementTree as ET

class Driver(object):
    def __init__(self, stage):
        self.stage  = stage
        self.parser = msgParser.MsgParser()
        self.state  = carState.CarState()
        self.control = carControl.CarControl()

        # -------- effectors ----------
        self.gear   = 1
        self.accel  = 0.0
        self.brake  = 0.0
        self.steer  = 0.0
        self.clutch = 0.0      # NEW
        self.meta   = 0        # NEW
        self.focus_angle = 0   # NEW  (requested focus direction)

        # -------- auxiliaries --------
        self.prev_gear = 1      # NEW – for clutch trigger
        self.last_focus_time = 0.0  # NEW – one focus / s
        self.stuck_timer = 0.0       # NEW – meta helper
        self.steering_scale = 1.0

        # track / car identification
        self.track_name, self.driver_module, self.car_model = self.get_track_driver_car()

        # -------- CSV logging --------
        file_exists = os.path.isfile("driving_data.csv")
        self.logfile = open("driving_data.csv", "a", newline="")
        self.logger  = csv.writer(self.logfile)

        if not file_exists:
            self.logger.writerow([
                "track","driver","car",
                # sensors
                "speedX","speedY","speedZ","angle","curLapTime","damage",
                "distFromStart","distRaced","focus","fuel","gear","lastLapTime",
                "opponents","racePos","rpm","track","trackPos","wheelSpinVel","z",
                # actuators
                "steering","accel","brake","clutch","meta","focusCmd",
                # keys
                "left_key","right_key","up_key","down_key","r_key"
            ])

    # -------------------------------------------------------------

    def get_track_driver_car(self):
        try:
            base_dir = r"C:\Users\ayaan\Documents\University\Semester 6\Aritificial Intelligence\Project\torcs"
            quickrace_xml = os.path.join(base_dir,"config/raceman/quickrace.xml")
            tree = ET.parse(quickrace_xml); root = tree.getroot()

            track_sec  = root.find(".//section[@name='Tracks']/section")
            track_name = track_sec.find("attstr[@name='name']").attrib['val'] if track_sec is not None else "unknown_track"

            drv_sec = root.find(".//section[@name='Drivers']")
            focused_idx  = drv_sec.find("attnum[@name='focused idx']").attrib['val'] if drv_sec is not None else "0"
            focused_mod  = drv_sec.find("attstr[@name='focused module']").attrib['val'] if drv_sec is not None else "unknown_driver"

            scr_xml   = os.path.join(base_dir,"drivers",focused_mod,"scr_server.xml")
            tree2 = ET.parse(scr_xml); root2 = tree2.getroot()
            driver_sec = root2.find(f".//section[@name='Robots']/section[@name='index']/section[@name='{focused_idx}']")
            if driver_sec is None:
                driver_sec = root2.find(".//section[@name='Robots']/section[@name='index']/section")
            car_name = driver_sec.find("attstr[@name='car name']").attrib['val'] if driver_sec is not None else "unknown_car"
            return track_name, focused_mod, car_name
        except Exception as e:
            print(f"[XML Error] {e}")
            return "unknown_track","unknown_driver","unknown_car"

    # -------------------------------------------------------------

    def init(self):
        angles = [0]*19
        for i in range(5):
            angles[i]          = -90 + i*15
            angles[18-i]       =  90 - i*15
        for i in range(5,9):
            angles[i]          = -20 + (i-5)*5
            angles[18-i]       =  20 - (i-5)*5
        return self.parser.stringify({'init': angles})

    # -------------------------------------------------------------

    def drive(self, msg):
        self.state.setFromMsg(msg)
        speedX = self.state.getSpeedX()
        rpm    = self.state.getRpm()
        curLap = self.state.getCurLapTime() or 0.0

        # ------- reset effectors every tick -------
        self.accel  = self.brake = self.steer = 0.0
        self.meta   = 0
        self.clutch = 0.0
        # focus_angle keeps last issued value

        # --------------------------------------------------
        #              KEYBOARD CONTROLS (unchanged)
        # --------------------------------------------------
        left_key  = 1 if keyboard.is_pressed('left')  else 0
        right_key = 1 if keyboard.is_pressed('right') else 0
        up_key    = 1 if keyboard.is_pressed('up')    else 0
        down_key  = 1 if keyboard.is_pressed('down')  else 0
        r_key     = 1 if keyboard.is_pressed('r')     else 0  # NEW manual restart

        target_steer =  self.steering_scale if left_key else (-self.steering_scale if right_key else 0.0)
        self.steer = 0.8*self.steer + 0.2*target_steer

        if down_key and speedX < 1:
            self.accel = 0.5; self.gear = -1
        elif up_key:
            self.accel = 1.0
            if self.gear < 1: self.gear = 1
        elif down_key:
            self.brake = 1.0

        if self.gear != -1:
            self.gear = min(6, max(1, int(speedX//50)+1))

        # -------------- simple track-edge dampening --------------
        track_pos = self.state.getTrackPos()
        if track_pos is not None and abs(track_pos) > 1.0:
            self.steer *= 0.5; self.accel = min(self.accel, 0.5)

        # --------------------------------------------------
        #        ***  NEW helpers for clutch / focus / meta  ***
        # --------------------------------------------------

        # clutch: press on any gear change for one tick
        if self.gear != self.prev_gear:
            self.clutch = 1.0
        self.prev_gear = self.gear

        # focus: once per simulated second, aim ±15° in steering direction
        if curLap - self.last_focus_time >= 1.0:
            self.focus_angle =  15 if self.steer >  0.05 else (-15 if self.steer < -0.05 else 0)
            self.last_focus_time = curLap
        self.control.setFocus(self.focus_angle)

        # meta: restart if stuck (>5 s, speed<2) or R key pressed
        if r_key:
            self.meta = 1
        elif speedX < 2:
            self.stuck_timer += 0.02
            if self.stuck_timer > 5:
                self.meta = 1
                self.stuck_timer = 0
        else:
            self.stuck_timer = 0

        # --------------------------------------------------
        #        SEND commands to TORCS
        # --------------------------------------------------
        self.control.setGear(self.gear)
        self.control.setAccel(self.accel)
        self.control.setBrake(self.brake)
        self.control.setSteer(self.steer)
        self.control.setClutch(self.clutch)     # NEW
        self.control.setMeta(self.meta)         # NEW

        # --------------------------------------------------
        #                LOGGING
        # --------------------------------------------------
        try:
            self.logger.writerow([
                self.track_name, self.driver_module, self.car_model,
                self.state.getSpeedX(), self.state.getSpeedY(), self.state.getSpeedZ(),
                self.state.getAngle(), self.state.getCurLapTime(), self.state.getDamage(),
                self.state.getDistFromStart(), self.state.getDistRaced(),
                str(self.state.getFocus()), self.state.getFuel(), self.state.getGear(),
                self.state.getLastLapTime(), str(self.state.getOpponents()),
                self.state.getRacePos(), rpm, str(self.state.getTrack()),
                self.state.getTrackPos(), str(self.state.getWheelSpinVel()),
                self.state.getZ(),
                # effectors
                self.steer, self.accel, self.brake, self.clutch, self.meta, self.focus_angle,
                # keys
                left_key, right_key, up_key, down_key, r_key
            ])
        except Exception as e:
            print(f"[Logging Error] {e}")

        print(f"[SEND] steer={self.steer:.2f} acc={self.accel:.2f} brk={self.brake:.2f} gear={self.gear} clutch={self.clutch} meta={self.meta}")
        return self.control.toMsg()

    # -------------------------------------------------------------

    def onShutDown(self):
        self.logfile.close()

    def onRestart(self):
        pass
