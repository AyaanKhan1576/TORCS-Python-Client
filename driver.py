
# # ───────── configuration ─────────
# USE_ML            = False      # False → keyboard / heuristic only
# FOCUS_ACTIVE      = False
# ENABLE_AUTO_RESET = False
# STUCK_SECONDS     = 15.0
# TARGET_SPEED_LOW  = 5.0        # m/s – below this we suppress ML brake

# MODEL_FILE   = "torcs_model.pt"
# PREPROC_FILE = "preproc.pkl"

# # ───────── imports ───────────────
# import os, csv, glob, re, math, xml.etree.ElementTree as ET
# import numpy as np, torch, keyboard, msgParser, carState, carControl
# from utils  import load_preproc, scale_row, ALL_NUM
# from models import build_mlp


# class Driver:
#     # ───────────────────────── init ──────────────────────────
#     def __init__(self, stage: int):
#         self.stage   = stage
#         self.parser  = msgParser.MsgParser()
#         self.state   = carState.CarState()
#         self.control = carControl.CarControl()

#         # effectors we still control
#         self.steer = self.accel = self.brake = 0.0
#         self.meta  = 0
#         self.focus_angle = 0
#         self.last_focus_time = 0.0
#         self.stuck_timer     = 0.0
#         self.steering_scale  = 1.0

#         # static info
#         self.track_name, self.car_model = self._get_track_and_car()

#         # optional ML model
#         self.model = None
#         if USE_ML and os.path.isfile(MODEL_FILE) and os.path.isfile(PREPROC_FILE):
#             self.scaler_stats, self.cat_maps = load_preproc(PREPROC_FILE)
#             emb = {k: max(m.values())+1 for k, m in self.cat_maps.items()}
#             self.model = build_mlp(len(ALL_NUM), emb)
#             self.model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
#             self.model.eval()
#             print("[ML] model loaded ✔")

#         # CSV logger
#         log_fn = self._make_log_filename()
#         self.logfile = open(log_fn, "a", newline="")
#         self.logger  = csv.writer(self.logfile)
#         if self.logfile.tell() == 0:
#             self.logger.writerow([
#                 "track","car",*ALL_NUM,
#                 "steering","accel","brake",
#                 "meta","focusCmd",
#                 "is_ml","left","right","up","down","r"
#             ])
#         print(f"[LOG] → {log_fn}")

#     # ───────────────────────── TORCS handshake ───────────────────────
#     def init(self):
#         ang = [0]*19
#         for i in range(5):   ang[i]    = -90+i*15;  ang[18-i] =  90-i*15
#         for i in range(5,9): ang[i]    = -20+(i-5)*5; ang[18-i] = 20-(i-5)*5
#         return self.parser.stringify({'init': ang})

#     # ───────────────────────── main loop ──────────────────────────────
#     def drive(self, msg):
#         s = self.state; s.setFromMsg(msg)
#         self.meta = 0

#         # 1) keyboard
#         kp = keyboard.is_pressed
#         left,right,up_key,down_key,rst = (kp('left'),kp('right'),
#                                           kp('up'),kp('down'),kp('r'))
#         manual_keys = any([left,right,up_key,down_key])
#         ml_active = USE_ML and self.model and not manual_keys

#         # 2) simulator sensors (gear & rpm come straight from TORCS)
#         sim_speedX = s.getSpeedX() or 0.0
#         sim_speedY = s.getSpeedY() or 0.0
#         sim_speedZ = s.getSpeedZ() or 0.0
#         sim_angle  = s.getAngle()  or 0.0
#         sim_trackPos = s.getTrackPos() or 0.0
#         sim_rpm   = s.getRpm()    or 0.0       # ← record only
#         sim_gear  = s.getGear()   or 0         # ← record only
#         sim_distRaced = s.getDistRaced() or 0.0
#         sim_damage    = s.getDamage()    or 0.0
#         sim_track     = s.getTrack()     or [0]*19
#         sim_focus     = s.getFocus()     or [0]*5
#         sim_wheelSpin = s.getWheelSpinVel() or [0]*4
#         sim_opps      = s.getOpponents()    or [0]*36

#         speed = math.sqrt(sim_speedX**2 + sim_speedY**2 + sim_speedZ**2)
#         track_pos = sim_trackPos

#         # 3) choose steering/throttle/brake (no clutch/gear logic)
#         if ml_active:
#             self.accel = self.brake = 0.0
#             sensors_for_ml = {
#                 **{ "speedX":sim_speedX,"speedY":sim_speedY,"speedZ":sim_speedZ,
#                     "angle":sim_angle,"trackPos":sim_trackPos,
#                     "rpm":sim_rpm,"gear":sim_gear,
#                     "distRaced":sim_distRaced,"damage":sim_damage},
#                 **{f"track{i}":v for i,v in enumerate(sim_track)},
#                 **{f"focus{i}":v for i,v in enumerate(sim_focus)},
#                 **{f"wheelSpinVel{i}":v for i,v in enumerate(sim_wheelSpin)},
#                 **{f"opponents{i}":v for i,v in enumerate(sim_opps)}
#             }
#             num = torch.from_numpy(np.asarray(
#                 [scale_row(sensors_for_ml, self.scaler_stats)], dtype=np.float32))
#             cat = torch.tensor([[ 
#                 self.cat_maps["track_name"].get(self.track_name,0),
#                 self.cat_maps["car_name"].get(self.car_model,0)
#             ]])
#             with torch.no_grad():
#                 st, ac, br = self.model(num, cat)[0].numpy()
#             if ac > br: br = 0.0
#             else:       ac = 0.0
#             if speed < TARGET_SPEED_LOW and br > 0.1: br = 0.0
#             self.steer = float(np.clip(st,-1,1))
#             self.accel = float(np.clip(ac,0,1))
#             self.brake = float(np.clip(br,0,1))
#         else:
#             self.steer = 0.8*self.steer + 0.2*(
#                 self.steering_scale if left else
#                 -self.steering_scale if right else 0.0)
#             if up_key:   self.accel = min(self.accel+0.03,1.0)
#             else:        self.accel = max(self.accel-0.10,0.0)
#             if down_key:
#                 self.brake = 1.0; self.accel = 0.0
#             else:
#                 self.brake = 0.0

#         # 4) focus rays (optional)
#         if FOCUS_ACTIVE:
#             lap = s.getCurLapTime() or 0.0
#             if lap - self.last_focus_time >= 1.0:
#                 self.focus_angle = 15 if self.steer>0.05 else -15 if self.steer<-0.05 else 0
#                 self.last_focus_time = lap
#         else:
#             self.focus_angle = 0
#         self.control.setFocus(self.focus_angle)

#         # 5) auto‐reset
#         if rst:
#             self.meta = 1
#         elif (ENABLE_AUTO_RESET and speed<0.5 and abs(track_pos)>1.2
#               and self.accel<0.05 and self.brake<0.05):
#             self.stuck_timer += 0.02
#             if self.stuck_timer >= STUCK_SECONDS:
#                 self.meta = 1; self.stuck_timer = 0.0
#         else:
#             self.stuck_timer = 0.0

#         # 6) send to TORCS (gear/clutch always 0 = “auto”)
#         self.control.setGear(0)         # let TORCS shift
#         self.control.setClutch(0.0)     # clutch free
#         self.control.setAccel(self.accel)
#         self.control.setBrake(self.brake)
#         self.control.setSteer(self.steer)
#         self.control.setMeta(self.meta)

#         # 7) log (gear & rpm straight from simulator)
#         self.logger.writerow([
#             self.track_name, self.car_model,
#             sim_speedX, sim_speedY, sim_speedZ,
#             sim_angle, sim_trackPos,
#             sim_rpm, sim_gear,                # ← exact values
#             sim_distRaced, sim_damage,
#             *[v for v in sim_track],
#             *[v for v in sim_focus],
#             *[v for v in sim_wheelSpin],
#             *[v for v in sim_opps],
#             self.steer, self.accel, self.brake,
#             self.meta, self.focus_angle,
#             int(ml_active),
#             int(left), int(right), int(up_key), int(down_key), int(rst)
#         ])
#         self.logfile.flush()

#         if self.meta: self.meta = 0
#         return self.control.toMsg()

#     # ───────── helpers ─────────
#     def onShutDown(self):
#         if self.logfile:
#             self.logfile.close()
#             print("[LOG] closed.")

#     def onRestart(self):
#         self.steer = self.accel = self.brake = 0.0
#         self.stuck_timer = 0.0
#         print("[Driver] Restarted.")

#     # def _get_track_and_car(self):
#     #     return "unknown_track", "unknown_car"

#     def _get_track_and_car(self):
#         try:
#             base = r"C:\Users\ayaan\Documents\University\Semester 6\Aritificial Intelligence\Project\torcs"
#             root = ET.parse(os.path.join(base, "config/raceman/quickrace.xml")).getroot()
#             track = root.find(".//section[@name='Tracks']/section/attstr[@name='name']").attrib['val']
#             drv   = root.find(".//section[@name='Drivers']")
#             idx   = drv.find("attnum[@name='focused idx']").attrib['val']
#             mod   = drv.find("attstr[@name='focused module']").attrib['val']
#             scr   = ET.parse(os.path.join(base, "drivers", mod, "scr_server.xml")).getroot()
#             rob   = scr.find(f".//section[@name='Robots']/section[@name='index']/section[@name='{idx}']") \
#                     or scr.find(".//section[@name='Robots']/section[@name='index']/section")
#             car   = rob.find("attstr[@name='car name']").attrib['val']
#             return track, car
#         except Exception:
#             return "unknown", "unknown_car"

#     def _make_log_filename(self):
#         slug = lambda x: re.sub(r"[^A-Za-z0-9_.-]", "", x)
#         t, c = slug(self.track_name), slug(self.car_model)
#         if t and c and t!="unknown_track" and c!="unknown_car":
#             base = f"{t}_{c}"
#             return f"{base}_{len(glob.glob(base+'_*'))+1:02}.csv"
#         return "driving_data.csv"

# driver.py  — hybrid TORCS client (ML + keyboard)  ⚙️
# ---------------------------------------------------------------------
#  • “speed // 50” gear logic restored (plus manual R/1st helpers).
#  • Full CSV rows now match the header: every numeric sensor in ALL_NUM,
#    followed by control/debug columns.
# ---------------------------------------------------------------------

# ───────── configuration ─────────
USE_ML            = False            # False → keyboard / heuristic only
FOCUS_ACTIVE      = False
ENABLE_AUTO_RESET = True
STUCK_SECONDS     = 15.0
TARGET_SPEED_LOW  = 5.0              # m/s – let ML brake only above this

MODEL_FILE   = "torcs_model.pt"
PREPROC_FILE = "preproc.pkl"

# ───────── imports ───────────────
import os, csv, glob, re, math
import numpy as np, torch, keyboard, msgParser, carState, carControl
import xml.etree.ElementTree as ET
from utils  import load_preproc, scale_row, ALL_NUM
from models import build_mlp


class Driver:
    # ───────────────── init ──────────────────
    def __init__(self, stage: int):
        # objects
        self.stage   = stage
        self.parser  = msgParser.MsgParser()
        self.state   = carState.CarState()
        self.control = carControl.CarControl()

        # effectors & helpers
        self.gear   = 1       # gear we command
        self.clutch = 0.0
        self.steer  = self.accel = self.brake = 0.0
        self.meta   = 0
        self.focus_angle   = 0
        self.last_focus_time = 0.0
        self.stuck_timer     = 0.0
        self.steering_scale  = 1.0

        # static info
        self.track_name, self.car_model = self._get_track_and_car()

        # optional ML model
        self.model = None
        if USE_ML and os.path.isfile(MODEL_FILE) and os.path.isfile(PREPROC_FILE):
            self.scaler_stats, self.cat_maps = load_preproc(PREPROC_FILE)
            emb = {k: max(m.values())+1 for k,m in self.cat_maps.items()}
            self.model = build_mlp(len(ALL_NUM), emb)
            self.model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
            self.model.eval()
            print("[ML] model loaded ✔")

        # CSV logger
        log_fn = self._make_log_filename()
        self.logfile = open(log_fn, "a", newline="")
        self.logger  = csv.writer(self.logfile)
        if self.logfile.tell() == 0:           # header only once
            self.logger.writerow([
                "track","car",*ALL_NUM,
                "steering","accel","brake",
                "clutch","meta","focusCmd",
                "is_ml","left","right","up","down","r",
                "commandedGear"
            ])
        print(f"[LOG] → {log_fn}")

    # ───────── handshake ──────────
    def init(self):
        ang=[0]*19
        for i in range(5):   ang[i]    = -90+i*15;  ang[18-i] =  90-i*15
        for i in range(5,9): ang[i]    = -20+(i-5)*5; ang[18-i] = 20-(i-5)*5
        return self.parser.stringify({'init': ang})

    # ───────── main loop ──────────
    def drive(self, msg):
        s = self.state; s.setFromMsg(msg)
        self.meta = 0

        # 1) keyboard
        kp = keyboard.is_pressed
        left,right,up_key,down_key,rst = (
            kp('left'), kp('right'), kp('up'), kp('down'), kp('r')
        )
        manual_keys = any([left,right,up_key,down_key])
        ml_active = USE_ML and self.model and not manual_keys

        # 2) simulator sensors (gear & rpm are TORCS-reported, untouched)
        sim_speedX = s.getSpeedX() or 0.0
        sim_speedY = s.getSpeedY() or 0.0
        sim_speedZ = s.getSpeedZ() or 0.0
        sim_angle  = s.getAngle()  or 0.0
        sim_trackPos = s.getTrackPos() or 0.0
        sim_rpm   = s.getRpm()    or 0.0
        sim_gear  = s.getGear()   or 0
        sim_distRaced = s.getDistRaced() or 0.0
        sim_damage    = s.getDamage()    or 0.0
        sim_track     = s.getTrack()     or [0]*19
        sim_focus     = s.getFocus()     or [0]*5
        sim_wheelSpin = s.getWheelSpinVel() or [0]*4
        sim_opps      = s.getOpponents()    or [0]*36

        speed = math.sqrt(sim_speedX**2 + sim_speedY**2 + sim_speedZ**2)
        track_pos = sim_trackPos

        # 3) gear & clutch logic ------------------------------------------
        cmd_gear = self.gear  # default keep

        # manual helpers (reverse / first)
        if down_key and sim_speedX < 1:
            self.accel = 0.5
            cmd_gear = -1
        elif up_key:
            self.accel = 1.0
            if cmd_gear < 1:
                cmd_gear = 1
        elif down_key:            # brake key
            self.brake = 1.0

        # auto forward shifting by speed
        if cmd_gear != -1:
            cmd_gear = min(6, max(1, int(sim_speedX // 50) + 1))

        # clutch engagement
        gear_change     = cmd_gear != self.gear
        neutral_to_first= (sim_gear == 0 and cmd_gear > 0)
        self.clutch = 1.0 if (gear_change or neutral_to_first) else 0.0
        self.gear   = cmd_gear
        # -----------------------------------------------------------------

        # 4) steering / throttle / brake
        if ml_active:
            self.accel = self.brake = 0.0
            ml_sensors = {
                **dict(speedX=sim_speedX,speedY=sim_speedY,speedZ=sim_speedZ,
                       angle=sim_angle,trackPos=sim_trackPos,
                       rpm=sim_rpm,gear=sim_gear,
                       distRaced=sim_distRaced,damage=sim_damage),
                **{f"track{i}":v for i,v in enumerate(sim_track)},
                **{f"focus{i}":v for i,v in enumerate(sim_focus)},
                **{f"wheelSpinVel{i}":v for i,v in enumerate(sim_wheelSpin)},
                **{f"opponents{i}":v for i,v in enumerate(sim_opps)}
            }
            num = torch.from_numpy(np.asarray(
                    [scale_row(ml_sensors,self.scaler_stats)],dtype=np.float32))
            cat = torch.tensor([[ self.cat_maps["track_name"].get(self.track_name,0),
                                  self.cat_maps["car_name"].get(self.car_model,0)]])
            with torch.no_grad():
                st, ac, br = self.model(num, cat)[0].numpy()
            if ac > br: br = 0.0
            else:       ac = 0.0
            if speed < TARGET_SPEED_LOW and br > 0.1: br = 0.0
            self.steer = float(np.clip(st,-1,1))
            self.accel = float(np.clip(ac,0,1))
            self.brake = float(np.clip(br,0,1))
        else:
            self.steer = 0.8*self.steer + 0.2*(
                self.steering_scale if left else
                -self.steering_scale if right else 0.0)
            if up_key:   self.accel = min(self.accel+0.03,1.0)
            else:        self.accel = max(self.accel-0.10,0.0)
            if down_key:
                self.brake = 1.0; self.accel = 0.0
            else:
                self.brake = 0.0

        # 5) focus rays
        if FOCUS_ACTIVE:
            lap = s.getCurLapTime() or 0.0
            if lap - self.last_focus_time >= 1.0:
                self.focus_angle = 15 if self.steer>0.05 else -15 if self.steer<-0.05 else 0
                self.last_focus_time = lap
        else:
            self.focus_angle = 0
        self.control.setFocus(self.focus_angle)

        # 6) auto-reset
        if rst:
            self.meta = 1
        elif (ENABLE_AUTO_RESET and speed<0.5 and abs(track_pos)>1.2
              and self.accel<0.05 and self.brake<0.05):
            self.stuck_timer += 0.02
            if self.stuck_timer >= STUCK_SECONDS:
                self.meta = 1; self.stuck_timer = 0.0
        else:
            self.stuck_timer = 0.0

        # 7) send to TORCS
        self.control.setGear(self.gear)
        self.control.setClutch(self.clutch)
        self.control.setAccel(self.accel)
        self.control.setBrake(self.brake)
        self.control.setSteer(self.steer)
        self.control.setMeta(self.meta)

        # 8) build full sensors dict for CSV
        sensors = {
            "speedX": sim_speedX, "speedY": sim_speedY, "speedZ": sim_speedZ,
            "angle":  sim_angle,  "trackPos": sim_trackPos,
            "rpm":    sim_rpm,    "gear":    sim_gear,
            "distRaced": sim_distRaced, "damage": sim_damage,
            **{f"track{i}":        v for i,v in enumerate(sim_track)},
            **{f"focus{i}":        v for i,v in enumerate(sim_focus)},
            **{f"wheelSpinVel{i}": v for i,v in enumerate(sim_wheelSpin)},
            **{f"opponents{i}":    v for i,v in enumerate(sim_opps)}
        }

        # 9) CSV write
        self.logger.writerow([
            self.track_name, self.car_model,
            *[sensors[c] for c in ALL_NUM],      # every numeric field
            self.steer, self.accel, self.brake,
            self.clutch, self.meta, self.focus_angle,
            int(ml_active),
            int(left), int(right), int(up_key), int(down_key), int(rst),
            self.gear                             # commandedGear
        ])
        self.logfile.flush()

        if self.meta: self.meta = 0
        return self.control.toMsg()

    # ───────── helpers ─────────
    def onShutDown(self):
        if self.logfile:
            self.logfile.close()
            print("[LOG] closed.")

    def onRestart(self):
        self.gear = 1
        self.steer = self.accel = self.brake = 0.0
        self.clutch = 0.0
        self.stuck_timer = 0.0
        print("[Driver] Restarted.")

    def _get_track_and_car(self):
        try:
            root = os.path.expanduser("~/torcs/config/raceman/quickrace.xml")
            root = ET.parse(root).getroot()
            track = root.find(".//section[@name='Tracks']/section/attstr[@name='name']").attrib['val']
            drv   = root.find(".//section[@name='Drivers']")
            idx   = drv.find("attnum[@name='focused idx']").attrib['val']
            mod   = drv.find("attstr[@name='focused module']").attrib['val']
            scr   = ET.parse(os.path.expanduser(f"~/torcs/drivers/{mod}/scr_server.xml")).getroot()
            rob   = scr.find(f".//section[@name='Robots']/section[@name='index']/section[@name='{idx}']") \
                    or scr.find(".//section[@name='Robots']/section[@name='index']/section")
            car   = rob.find("attstr[@name='car name']").attrib['val']
            return track, car
        except Exception:
            return "unknown_track", "unknown_car"

    def _make_log_filename(self):
        slug = lambda x: re.sub(r"[^A-Za-z0-9_.-]", "", x)
        t, c = slug(self.track_name), slug(self.car_model)
        if t and c and t!="unknown_track" and c!="unknown_car":
            base = f"{t}_{c}"
            return f"{base}_{len(glob.glob(base+'_*'))+1:02}.csv"
        return "driving_data.csv"
