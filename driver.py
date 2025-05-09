# import msgParser, carState, carControl, keyboard
# import csv, os, time, xml.etree.ElementTree as ET
# import numpy as np, joblib

# MODEL_PATH = "torcs_model.pkl"   # ← must match train_model.py

# class Driver(object):
#     # ──────────────────────────────────────────────────────────
#     def __init__(self, stage):
#         self.stage  = stage
#         self.parser = msgParser.MsgParser()
#         self.state  = carState.CarState()
#         self.control = carControl.CarControl()

#         # effectors
#         self.gear = 1; self.accel = self.brake = self.steer = 0.0
#         self.clutch = 0.0; self.meta = 0; self.focus_angle = 0

#         # helpers
#         self.prev_gear = 1; self.last_focus_time = 0.0
#         self.stuck_timer = 0.0; self.steering_scale = 1.0

#         # --- ML model ---------------------------------------------------------
#         self.model = None
#         if os.path.isfile(MODEL_PATH):
#             try:
#                 self.model = joblib.load(MODEL_PATH)
#                 print(f"[ML] Loaded model: {MODEL_PATH}")
#             except Exception as e:
#                 print(f"[ML] Could not load model ({e}) – fallback to keyboard.")

#         # track / car info
#         self.track_name, self.driver_module, self.car_model = self.get_track_driver_car()

#         # --- CSV logger -------------------------------------------------------
#         file_exists = os.path.isfile("driving_data.csv")
#         self.logfile = open("driving_data.csv", "a", newline="")
#         self.logger  = csv.writer(self.logfile)
#         if not file_exists:
#             self.logger.writerow([
#                 "track","driver","car",
#                 "speedX","speedY","speedZ","angle","curLapTime","damage",
#                 "distFromStart","distRaced","focus","fuel","gear","lastLapTime",
#                 "opponents","racePos","rpm","track","trackPos","wheelSpinVel","z",
#                 "steering","accel","brake","clutch","meta","focusCmd",
#                 "left_key","right_key","up_key","down_key","r_key"
#             ])

#     # ──────────────────────────────────────────────────────────
#     def get_track_driver_car(self):
#         try:
#             base = r"C:\Users\ayaan\Documents\University\Semester 6\Aritificial Intelligence\Project\torcs"
#             qxml = os.path.join(base, "config/raceman/quickrace.xml")
#             t = ET.parse(qxml); r = t.getroot()

#             track = r.find(".//section[@name='Tracks']/section")
#             track_name = track.find("attstr[@name='name']").attrib['val'] if track is not None else "unknown"

#             dsec = r.find(".//section[@name='Drivers']")
#             idx  = dsec.find("attnum[@name='focused idx']").attrib['val'] if dsec is not None else "0"
#             mod  = dsec.find("attstr[@name='focused module']").attrib['val'] if dsec is not None else "unknown"

#             scr = os.path.join(base, "drivers", mod, "scr_server.xml")
#             r2  = ET.parse(scr).getroot()
#             rob = r2.find(f".//section[@name='Robots']/section[@name='index']/section[@name='{idx}']")
#             if rob is None:
#                 rob = r2.find(".//section[@name='Robots']/section[@name='index']/section")
#             car = rob.find("attstr[@name='car name']").attrib['val'] if rob is not None else "unknown_car"
#             return track_name, mod, car
#         except Exception as e:
#             print(f"[XML] {e}")
#             return "unknown","unknown","unknown"

#     # ──────────────────────────────────────────────────────────
#     def init(self):
#         ang = [0]*19
#         for i in range(5):
#             ang[i] = -90+i*15; ang[18-i] = 90-i*15
#         for i in range(5,9):
#             ang[i] = -20+(i-5)*5; ang[18-i] = 20-(i-5)*5
#         return self.parser.stringify({'init': ang})

#     # ──────────────────────────────────────────────────────────
#     def drive(self, msg):
#         self.state.setFromMsg(msg)

#         # ---------------- sensor shortcuts -------------------
#         speedX  = self.state.getSpeedX() or 0
#         angle   = self.state.getAngle()  or 0
#         trackP  = self.state.getTrackPos() or 0
#         rpm     = self.state.getRpm()    or 0
#         curLap  = self.state.getCurLapTime() or 0.0

#         # ---------------- keyboard override ------------------
#         l = keyboard.is_pressed
#         left_k  = 1 if l('left')  else 0
#         right_k = 1 if l('right') else 0
#         up_k    = 1 if l('up')    else 0
#         down_k  = 1 if l('down')  else 0
#         r_k     = 1 if l('r')     else 0

#         # ------------------------------------------------------------------
#         #   1)  get ML prediction
#         # ------------------------------------------------------------------
#         if self.model is not None and not any([left_k,right_k,up_k,down_k]):
#             feat = np.array([[speedX, angle, trackP, rpm]])
#             steer_pred, accel_pred, brake_pred = self.model.predict(feat)[0]
#             # clamp to legal ranges
#             self.steer  = float(np.clip(steer_pred, -1, 1))
#             self.accel  = float(np.clip(accel_pred,  0, 1))
#             self.brake  = float(np.clip(brake_pred,  0, 1))
#         else:
#             # ----------------------------------------------------------------
#             #   2)  manual / simple logic (same as before)
#             # ----------------------------------------------------------------
#             target_steer = self.steering_scale if left_k else (-self.steering_scale if right_k else 0.0)
#             self.steer   = 0.8*self.steer + 0.2*target_steer

#             if down_k and speedX < 1:
#                 self.accel = 0.5; self.gear = -1
#             elif up_k:
#                 self.accel = 1.0
#                 if self.gear < 1: self.gear = 1
#             elif down_k:
#                 self.brake = 1.0

#         # ---------------- basic speed→gear heuristic -------------
#         if self.gear != -1:
#             self.gear = min(6, max(1, int(speedX//50)+1))

#         # ---------------- clutch on gear change -----------------
#         self.clutch = 1.0 if self.gear != self.prev_gear else 0.0
#         self.prev_gear = self.gear

#         # ---------------- focus sensor sweep -------------------
#         if curLap - self.last_focus_time >= 1.0:
#             self.focus_angle =  15 if self.steer > 0.05 else (-15 if self.steer < -0.05 else 0)
#             self.last_focus_time = curLap
#         self.control.setFocus(self.focus_angle)

#         # ---------------- auto restart -------------------------
#         if r_k:
#             self.meta = 1
#         elif speedX < 2:
#             self.stuck_timer += 0.02
#             if self.stuck_timer > 5:
#                 self.meta = 1; self.stuck_timer = 0
#         else: self.stuck_timer = 0

#         # ---------------- send to TORCS ------------------------
#         self.control.setGear(self.gear)
#         self.control.setAccel(self.accel)
#         self.control.setBrake(self.brake)
#         self.control.setSteer(self.steer)
#         self.control.setClutch(self.clutch)
#         self.control.setMeta(self.meta)

#         # ---------------- CSV log ------------------------------
#         try:
#             self.logger.writerow([
#                 self.track_name, self.driver_module, self.car_model,
#                 self.state.getSpeedX(), self.state.getSpeedY(), self.state.getSpeedZ(),
#                 angle, self.state.getCurLapTime(), self.state.getDamage(),
#                 self.state.getDistFromStart(), self.state.getDistRaced(),
#                 str(self.state.getFocus()), self.state.getFuel(), self.state.getGear(),
#                 self.state.getLastLapTime(), str(self.state.getOpponents()),
#                 self.state.getRacePos(), rpm, str(self.state.getTrack()),
#                 trackP, str(self.state.getWheelSpinVel()), self.state.getZ(),
#                 self.steer, self.accel, self.brake, self.clutch, self.meta, self.focus_angle,
#                 left_k, right_k, up_k, down_k, r_k
#             ])
#         except Exception as e:
#             print(f"[Log] {e}")

#         print(f"[SEND] st={self.steer:+.2f} ac={self.accel:.2f} br={self.brake:.2f} g={self.gear} ml={'Y' if self.model else 'N'}")
#         return self.control.toMsg()

#     # ──────────────────────────────────────────────────────────
#     def onShutDown(self):
#         self.logfile.close()

#     def onRestart(self): pass




# # driver.py ------------------------------------------------------------
# # TORCS client that loads torch model with rich sensor set
# import msgParser, carState, carControl, keyboard
# import csv, os, xml.etree.ElementTree as ET, numpy as np, torch
# from utils import load_preproc, scale_row, ALL_NUM

# MODEL_FN   = "torcs_model.pt"
# PREPROC_FN = "preproc.pkl"

# class Driver(object):
#     # ──────────────────────────────────────────────────────────
#     def __init__(self, stage):
#         self.stage   = stage
#         self.parser  = msgParser.MsgParser()
#         self.state   = carState.CarState()
#         self.control = carControl.CarControl()

#         # effectors
#         self.gear = 1; self.accel = self.brake = self.steer = 0.0
#         self.clutch = 0.0; self.meta = 0; self.focus_angle = 0

#         # helpers
#         self.prev_gear = 1; self.last_focus_time = 0.0
#         self.stuck_timer = 0.0; self.steering_scale = 1.0

#         # load track / car labels
#         self.track_name, self.driver_module, self.car_model = self._get_track_info()

#         # ----- ML model ------------------------------------------------------
#         self.model, self.scaler_stats, self.cat_maps = None, None, None
#         if os.path.isfile(MODEL_FN) and os.path.isfile(PREPROC_FN):
#             from train_torch import MLP
#             try:
#                 self.scaler_stats, self.cat_maps = load_preproc(PREPROC_FN)
#                 self.model = MLP()
#                 self.model.load_state_dict(torch.load(MODEL_FN, map_location="cpu"))
#                 self.model.eval()
#                 print("[ML] neural model loaded")
#             except Exception as e:
#                 print("[ML] load error:", e)

#         # ----- CSV logging ---------------------------------------------------
#         fexists = os.path.isfile("driving_data.csv")
#         self.logfile = open("driving_data.csv", "a", newline="")
#         self.logger  = csv.writer(self.logfile)
#         if not fexists:
#             self.logger.writerow([
#                 "track","driver","car",
#                 *ALL_NUM,
#                 "steering","accel","brake","clutch","meta","focusCmd",
#                 "left_key","right_key","up_key","down_key","r_key"
#             ])

#     # ──────────────────────────────────────────────────────────
#     def _get_track_info(self):
#         try:
#             base = r"C:\Users\ayaan\Documents\University\Semester 6\Aritificial Intelligence\Project\torcs"
#             qxml = os.path.join(base, "config/raceman/quickrace.xml")
#             root = ET.parse(qxml).getroot()
#             tsec = root.find(".//section[@name='Tracks']/section")
#             track = tsec.find("attstr[@name='name']").attrib['val'] if tsec is not None else "unknown"
#             dsec = root.find(".//section[@name='Drivers']")
#             idx  = dsec.find("attnum[@name='focused idx']").attrib['val'] if dsec is not None else "0"
#             mod  = dsec.find("attstr[@name='focused module']").attrib['val'] if dsec is not None else "unknown"
#             scr  = os.path.join(base,"drivers",mod,"scr_server.xml")
#             r2   = ET.parse(scr).getroot()
#             rob  = r2.find(f".//section[@name='Robots']/section[@name='index']/section[@name='{idx}']") \
#                    or r2.find(".//section[@name='Robots']/section[@name='index']/section")
#             car  = rob.find("attstr[@name='car name']").attrib['val'] if rob is not None else "unknown_car"
#             return track, mod, car
#         except Exception as e:
#             print("[XML]", e); return "unknown","unknown","unknown"

#     # ──────────────────────────────────────────────────────────
#     def init(self):
#         ang = [0]*19
#         for i in range(5):
#             ang[i] = -90+i*15; ang[18-i] = 90-i*15
#         for i in range(5,9):
#             ang[i] = -20+(i-5)*5; ang[18-i] = 20-(i-5)*5
#         return self.parser.stringify({'init': ang})

#     # ──────────────────────────────────────────────────────────
#     def drive(self, msg):
#         self.state.setFromMsg(msg)

#         # ----- gather sensor dict -------------------------------------------
#         s = self.state
#         sensors = {
#             "speedX": s.getSpeedX() or 0.0,
#             "speedY": s.getSpeedY() or 0.0,
#             "speedZ": s.getSpeedZ() or 0.0,
#             "angle":  s.getAngle()  or 0.0,
#             "trackPos": s.getTrackPos() or 0.0,
#             "rpm":    s.getRpm()    or 0.0,
#             "gear":   s.getGear()   or 0,
#             "distRaced": s.getDistRaced() or 0.0,
#             "damage": s.getDamage() or 0.0,
#             **{f"track{i}": v for i,v in enumerate((s.getTrack() or [0]*19))},
#             **{f"focus{i}": v for i,v in enumerate((s.getFocus() or [0]*5))},
#             **{f"wheelSpinVel{i}": v for i,v in enumerate((s.getWheelSpinVel() or [0]*4))},
#             **{f"opponents{i}": v for i,v in enumerate((s.getOpponents() or [0]*36))}
#         }

#         # ----- keyboard override keys ---------------------------------------
#         k = keyboard.is_pressed
#         left_k  = 1 if k('left')  else 0
#         right_k = 1 if k('right') else 0
#         up_k    = 1 if k('up')    else 0
#         down_k  = 1 if k('down')  else 0
#         r_k     = 1 if k('r')     else 0
#         manual  = any([left_k,right_k,up_k,down_k])

#         # --------------------------------------------------------------------
#         if self.model and not manual:
#             num = torch.tensor([scale_row(sensors, self.scaler_stats)])
#             cat = torch.tensor([[ self.cat_maps["track_name"].get(self.track_name,0),
#                                   self.cat_maps["car_name"].get(self.car_model,0) ]])
#             with torch.no_grad():
#                 steer, accel, brake = self.model(num, cat)[0].numpy().tolist()
#             self.steer = float(np.clip(steer,-1,1))
#             self.accel = float(np.clip(accel,0,1))
#             self.brake = float(np.clip(brake,0,1))
#         else:
#             tgt = self.steering_scale if left_k else (-self.steering_scale if right_k else 0.0)
#             self.steer = 0.8*self.steer + 0.2*tgt
#             if down_k and sensors["speedX"] < 1:
#                 self.accel, self.gear = 0.5, -1
#             elif up_k:
#                 self.accel = 1.0; self.gear = max(1, self.gear)
#             elif down_k: self.brake = 1.0

#         # ----- simple gear heuristic & clutch -------------------------------
#         if self.gear != -1:
#             self.gear = min(6, max(1, int(sensors["speedX"]//50)+1))
#         self.clutch = 1.0 if self.gear != self.prev_gear else 0.0
#         self.prev_gear = self.gear

#         # ----- focus ray sweep ----------------------------------------------
#         curLap = s.getCurLapTime() or 0.0
#         if curLap - self.last_focus_time >= 1.0:
#             self.focus_angle =  15 if self.steer > 0.05 else (-15 if self.steer < -0.05 else 0)
#             self.last_focus_time = curLap
#         self.control.setFocus(self.focus_angle)

#         # ----- meta restart (stuck or 'r') ----------------------------------
#         if r_k: self.meta = 1
#         elif sensors["speedX"] < 2:
#             self.stuck_timer += 0.02
#             if self.stuck_timer > 5: self.meta, self.stuck_timer = 1, 0
#         else: self.stuck_timer = 0

#         # ----- send to TORCS -------------------------------------------------
#         self.control.setGear(self.gear)
#         self.control.setAccel(self.accel); self.control.setBrake(self.brake)
#         self.control.setSteer(self.steer); self.control.setClutch(self.clutch)
#         self.control.setMeta(self.meta)

#         # ----- logging -------------------------------------------------------
#         try:
#             self.logger.writerow([
#                 self.track_name, self.driver_module, self.car_model,
#                 *[sensors[c] for c in ALL_NUM],
#                 self.steer, self.accel, self.brake, self.clutch, self.meta, self.focus_angle,
#                 left_k, right_k, up_k, down_k, r_k
#             ])
#         except Exception as e:
#             print("[Log]", e)

#         print(f"[SEND] st={self.steer:+.2f} ac={self.accel:.2f} br={self.brake:.2f} "
#               f"g={self.gear} ml={'Y' if self.model and not manual else 'N'}")
#         return self.control.toMsg()

#     # ──────────────────────────────────────────────────────────
#     def onShutDown(self): self.logfile.close()
#     def onRestart(self):  pass


# driver.py -------------------------------------------------------------------
# SET THIS FLAG to choose default behaviour
USE_ML = False          # True = let neural model drive by default
                       # False = heuristic/manual only

import os, csv, glob, re, xml.etree.ElementTree as ET
import numpy as np, torch, keyboard, msgParser, carState, carControl
from utils import load_preproc, scale_row, ALL_NUM

MODEL_FILE   = "torcs_model.pt"
PREPROC_FILE = "preproc.pkl"


class Driver(object):

    # ──────────────────────────────────────────────────────────────────
    def __init__(self, stage):
        # core objects
        self.stage = stage
        self.parser = msgParser.MsgParser()
        self.state  = carState.CarState()
        self.control = carControl.CarControl()

        # effectors
        self.gear = 1
        self.steer = self.accel = self.brake = 0.0
        self.clutch = 0.0
        self.meta   = 0
        self.focus_angle = 0

        # helpers
        self.prev_gear = 1
        self.last_focus_time = 0.0
        self.stuck_timer = 0.0
        self.steering_scale = 1.0

        # track & car info
        self.track_name, self.car_model = self._get_track_and_car()

        # ML model
        self.model = None
        self.scaler_stats = None
        self.cat_maps = None
        if USE_ML and os.path.isfile(MODEL_FILE) and os.path.isfile(PREPROC_FILE):
            from train_model import MLP  # reuse class
            self.scaler_stats, self.cat_maps = load_preproc(PREPROC_FILE)
            self.model = MLP()
            self.model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
            self.model.eval()
            print("[ML] Model loaded ✔")

        # CSV logger
        csv_name = self._make_log_filename()
        self.logfile = open(csv_name, "a", newline="")
        self.logger  = csv.writer(self.logfile)
        if self.logfile.tell() == 0:
            self.logger.writerow([
                "track", "car", *ALL_NUM,
                "steering", "accel", "brake",
                "clutch", "meta", "focusCmd",
                "is_ml", "left_key", "right_key", "up_key", "down_key", "r_key"
            ])
        print(f"[LOG] Writing to {csv_name}")

    # ──────────────────────────────────────────────────────────────────
    # TORCS required init sensor angles
    def init(self):
        ang = [0] * 19
        for i in range(5):
            ang[i] = -90 + i * 15
            ang[18 - i] = 90 - i * 15
        for i in range(5, 9):
            ang[i] = -20 + (i - 5) * 5
            ang[18 - i] = 20 - (i - 5) * 5
        return self.parser.stringify({'init': ang})

    # ──────────────────────────────────────────────────────────────────
    # Main driving loop
    def drive(self, msg):

        # ---------- set sensors ----------
        s = self.state
        s.setFromMsg(msg)
        sen = {
            "speedX": s.getSpeedX() or 0.0,
            "speedY": s.getSpeedY() or 0.0,
            "speedZ": s.getSpeedZ() or 0.0,
            "angle":  s.getAngle()  or 0.0,
            "trackPos": s.getTrackPos() or 0.0,
            "rpm":    s.getRpm()    or 0.0,
            "gear":   s.getGear()   or 0,
            "distRaced": s.getDistRaced() or 0.0,
            "damage": s.getDamage() or 0.0,
            **{f"track{i}": v for i, v in enumerate(s.getTrack() or [0]*19)},
            **{f"focus{i}": v for i, v in enumerate(s.getFocus() or [0]*5)},
            **{f"wheelSpinVel{i}": v for i, v in enumerate(s.getWheelSpinVel() or [0]*4)},
            **{f"opponents{i}": v for i, v in enumerate(s.getOpponents() or [0]*36)}
        }

        # ---------- keyboard ----------
        k = keyboard.is_pressed
        l, r, u, d, rst = (k('left'), k('right'), k('up'),
                           k('down'), k('r'))
        manual_override = any([l, r, u, d])
        ml_active = USE_ML and self.model and not manual_override

        # ---------- control ----------
        if ml_active:
            num = torch.tensor([scale_row(sen, self.scaler_stats)])
            cat = torch.tensor([[self.cat_maps["track_name"].get(self.track_name, 0),
                                 self.cat_maps["car_name"].get(self.car_model, 0)]])
            with torch.no_grad():
                self.steer, self.accel, self.brake = self.model(num, cat)[0].numpy().tolist()
            self.steer = float(np.clip(self.steer, -1, 1))
            self.accel = float(np.clip(self.accel, 0, 1))
            self.brake = float(np.clip(self.brake, 0, 1))
        else:
            tgt = self.steering_scale if l else (-self.steering_scale if r else 0.0)
            self.steer = 0.8 * self.steer + 0.2 * tgt
            if d and sen["speedX"] < 1:
                self.accel, self.gear = 0.5, -1
            elif u:
                self.accel = 1.0
                self.gear = max(1, self.gear)
            elif d:
                self.brake = 1.0

        # ---------- gear & clutch ----------
        if self.gear != -1:
            self.gear = min(6, max(1, int(sen["speedX"] // 50) + 1))
        self.clutch = 1.0 if self.gear != self.prev_gear else 0.0
        self.prev_gear = self.gear

        # ---------- focus sweep ----------
        lap = s.getCurLapTime() or 0.0
        if lap - self.last_focus_time >= 1.0:
            self.focus_angle = 15 if self.steer > 0.05 else -15 if self.steer < -0.05 else 0
            self.last_focus_time = lap
        self.control.setFocus(self.focus_angle)

        # ---------- meta restart ----------
        if rst:
            self.meta = 1
        elif sen["speedX"] < 2:
            self.stuck_timer += 0.02
        else:
            self.stuck_timer = 0.0
        if self.stuck_timer > 5:
            self.meta = 1
            self.stuck_timer = 0.0

        # ---------- send actions ----------
        self.control.setGear(self.gear)
        self.control.setAccel(self.accel)
        self.control.setBrake(self.brake)
        self.control.setSteer(self.steer)
        self.control.setClutch(self.clutch)
        self.control.setMeta(self.meta)

        # ---------- log ----------
        self.logger.writerow([
            self.track_name, self.car_model,
            *[sen[c] for c in ALL_NUM],
            self.steer, self.accel, self.brake,
            self.clutch, self.meta, self.focus_angle,
            int(ml_active), int(l), int(r), int(u), int(d), int(rst)
        ])

        return self.control.toMsg()

    # ──────────────────────────────────────────────────────────────────
    def onShutDown(self):
        self.logfile.close()

    def onRestart(self):
        pass

    # ──────────────────────────────────────────────────────────────────
    # helper: parse XML to find track & car
    def _get_track_and_car(self):
        try:
            base = r"C:\Users\ayaan\Documents\University\Semester 6\Aritificial Intelligence\Project\torcs"
            root = ET.parse(os.path.join(base, "config/raceman/quickrace.xml")).getroot()
            track = root.find(".//section[@name='Tracks']/section/attstr[@name='name']").attrib['val']
            drv   = root.find(".//section[@name='Drivers']")
            idx   = drv.find("attnum[@name='focused idx']").attrib['val']
            mod   = drv.find("attstr[@name='focused module']").attrib['val']
            scr   = ET.parse(os.path.join(base, "drivers", mod, "scr_server.xml")).getroot()
            rob   = scr.find(f".//section[@name='Robots']/section[@name='index']/section[@name='{idx}']") \
                    or scr.find(".//section[@name='Robots']/section[@name='index']/section")
            car   = rob.find("attstr[@name='car name']").attrib['val']
            return track, car
        except Exception:
            return "unknown", "unknown_car"

    # name like  <Track>_<Car>_<N>.csv
    def _make_log_filename(self):
        def slug(x): return re.sub("[^A-Za-z0-9]", "", x)
        t, c = slug(self.track_name), slug(self.car_model)
        if t and c and t != "unknown" and c != "unknowncar":
            base = f"{t}_{c}"
            idx  = len(glob.glob(f"{base}_*.csv")) + 1
            return f"{base}_{idx:02}.csv"
        return "driving_data.csv"
