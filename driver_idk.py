
# # ─────────────────────────── configuration ───────────────────────────
# USE_ML            = False      # set True to use torcs_model.pt + preproc.pkl
# FOCUS_ACTIVE      = False
# ENABLE_AUTO_RESET = True
# STUCK_SECONDS     = 15.0
# TARGET_SPEED_LOW  = 5.0        # m/s – below this we suppress ML brake

# MODEL_FILE   = "torcs_model.pt"
# PREPROC_FILE = "preproc.pkl"

# # ───────────────────────────── imports ───────────────────────────────
# import os, csv, glob, re, math, time
# import numpy as np, torch, keyboard, msgParser, carState, carControl
# from utils  import load_preproc, scale_row, ALL_NUM
# from models import build_mlp

# # ──────────────────────────── Driver class ───────────────────────────
# class Driver:
#     # ──────────────────────────── init ───────────────────────────────
#     def __init__(self, stage):
#         self.stage   = stage
#         self.parser  = msgParser.MsgParser()
#         self.state   = carState.CarState()
#         self.control = carControl.CarControl()

#         # effectors & helpers
#         self.gear = 1
#         self.prev_gear = 1
#         self.steer = self.accel = self.brake = 0.0
#         self.clutch = 0.0; self.meta = 0
#         self.focus_angle = 0
#         self.stuck_timer = 0.0; self.last_focus_time = 0.0
#         self.steering_scale = 1.0

#         # ── new gear-tracking helpers (from reference) ──
#         self.last_shift_time = 0.0
#         self.last_gear_up    = False
#         self.last_gear_down  = False

#         # static info (optional names)
#         self.track_name, self.car_model = "unknown_track", "unknown_car"

#         # optional torch MLP
#         self.model = None
#         if USE_ML and os.path.isfile(MODEL_FILE) and os.path.isfile(PREPROC_FILE):
#             self.scaler_stats, self.cat_maps = load_preproc(PREPROC_FILE)
#             emb = {k: max(m.values())+1 for k, m in self.cat_maps.items()}
#             self.model = build_mlp(len(ALL_NUM), emb)
#             self.model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
#             self.model.eval()
#             print("[ML] model loaded ✓")

#         # CSV logger
#         log_fn = self._make_log_filename()
#         self.logfile = open(log_fn, "a", newline="")
#         self.logger  = csv.writer(self.logfile)
#         if self.logfile.tell() == 0:
#             self.logger.writerow([
#                 "track","car",*ALL_NUM,
#                 "steering","accel","brake",
#                 "clutch","meta","focusCmd",
#                 "is_ml","left","right","up","down","r"
#             ])
#         print(f"[LOG] → {log_fn}")

#     # ─────────────────── handshake (angles) ───────────────────────────
#     def init(self):
#         ang=[0]*19
#         for i in range(5):  ang[i]=-90+i*15; ang[18-i]= 90-i*15
#         for i in range(5,9):ang[i]=-20+(i-5)*5; ang[18-i]= 20-(i-5)*5
#         return self.parser.stringify({'init': ang})

#     # ─────────────────────────── main loop ────────────────────────────
#     def drive(self, msg):
#         s = self.state;  s.setFromMsg(msg);  self.meta = 0
#         now = time.time()

#         # ---------- keyboard states ----------
#         k_left  = keyboard.is_pressed('a') or keyboard.is_pressed('left')
#         k_right = keyboard.is_pressed('d') or keyboard.is_pressed('right')
#         k_throt = keyboard.is_pressed('w') or keyboard.is_pressed('up')
#         k_brake = keyboard.is_pressed('s') or keyboard.is_pressed('down')
#         k_rst   = keyboard.is_pressed('r')
#         manual_any = k_left or k_right or k_throt or k_brake

#         # ---------- sensors ----------
#         sim_speedX = s.getSpeedX() or 0.0   # km/h
#         sim_speedY = s.getSpeedY() or 0.0
#         sim_speedZ = s.getSpeedZ() or 0.0
#         sim_rpm    = s.getRpm()    or 0.0
#         track_pos  = s.getTrackPos() or 0.0
#         speed_mag  = math.sqrt(sim_speedX*sim_speedX +
#                                sim_speedY*sim_speedY +
#                                sim_speedZ*sim_speedZ)

#         # ---------- steering + pedals ----------
#         if USE_ML and self.model and not manual_any:
#             feat = {c:0.0 for c in ALL_NUM}
#             feat.update({
#                 "speedX":sim_speedX,"speedY":sim_speedY,"speedZ":sim_speedZ,
#                 "angle": s.getAngle() or 0.0,
#                 "trackPos": track_pos,
#                 "rpm": sim_rpm,
#                 "gear": self.gear,
#                 "distRaced": s.getDistRaced() or 0.0,
#                 "damage": s.getDamage() or 0.0,
#             })
#             for i,v in enumerate(s.getTrack() or [0]*19):   feat[f"track{i}"]=v
#             for i,v in enumerate(s.getFocus() or [0]*5):    feat[f"focus{i}"]=v
#             for i,v in enumerate(s.getWheelSpinVel() or [0]*4): feat[f"wheelSpinVel{i}"]=v
#             for i,v in enumerate(s.getOpponents() or [0]*36):   feat[f"opponents{i}"]=v

#             num = torch.from_numpy(np.asarray([scale_row(feat,self.scaler_stats)],
#                                               dtype=np.float32))
#             cat = torch.tensor([[ self.cat_maps["track_name"].get(self.track_name,0),
#                                   self.cat_maps["car_name"].get(self.car_model,0) ]])
#             with torch.no_grad():
#                 steer, accel, brake = self.model(num, cat)[0].numpy()

#             if accel > brake: brake = 0.0
#             else: accel = 0.0
#             if sim_speedX < TARGET_SPEED_LOW and brake > 0.1: brake = 0.0

#             self.steer = float(np.clip(steer, -1, 1))
#             self.accel = float(np.clip(accel, 0, 1))
#             self.brake = float(np.clip(brake, 0, 1))
#             ml_flag = 1
#         else:
#             # keyboard heuristic
#             self.steer = 0.8*self.steer + 0.2*( 0.8 if k_left else -0.8 if k_right else 0 )
#             self.accel = 1.0 if k_throt else 0.0
#             self.brake = 1.0 if k_brake else 0.0
#             ml_flag = 0

#         # ────────────────── GEARBOX (reference logic) ─────────────────
#         rpm      = sim_rpm
#         speedX_kmh = sim_speedX

#         # --- manual ↑ / ↓ (single press) ---
#         if keyboard.is_pressed('up'):
#             if not self.last_gear_up:
#                 self.gear = min(self.gear + 1, 7)
#                 self.last_shift_time = now
#                 self.last_gear_up = True
#         else:
#             self.last_gear_up = False

#         if keyboard.is_pressed('down'):
#             if not self.last_gear_down:
#                 self.gear = max(self.gear - 1, 1)
#                 self.last_shift_time = now
#                 self.last_gear_down = True
#         else:
#             self.last_gear_down = False

#         # --- auto-shift with 0.5 s cooldown ---
#         if now - self.last_shift_time > 0.5:
#             if rpm > 8000 and self.gear < 7:
#                 self.gear += 1
#                 self.last_shift_time = now
#             elif rpm < 4000 and self.gear > 1:
#                 self.gear -= 1
#                 self.last_shift_time = now

#         # --- direction sanity ---
#         if speedX_kmh < 0 and self.gear > 0:
#             self.gear = -1
#         elif speedX_kmh > 0 and self.gear < 0:
#             self.gear = 1

#         # clamp & skip neutral
#         self.gear = max(min(self.gear, 7), -1)
#         if self.gear == 0: self.gear = 1

#         # clutch engages on change
#         self.clutch = 1.0 if self.gear != self.prev_gear else 0.0
#         self.prev_gear = self.gear
#         # ──────────────────────────────────────────────────────────────

#         # ---------- focus rays (optional) ----------
#         if FOCUS_ACTIVE:
#             lap = s.getCurLapTime() or 0.0
#             if lap - self.last_focus_time >= 1.0:
#                 self.focus_angle = 15 if self.steer>0.05 else -15 if self.steer<-0.05 else 0
#                 self.last_focus_time = lap
#         else:
#             self.focus_angle = 0
#         self.control.setFocus(self.focus_angle)

#         # ---------- auto-reset ----------
#         if k_rst:
#             self.meta = 1
#         elif (ENABLE_AUTO_RESET and speed_mag<0.5 and abs(track_pos)>1.2
#               and self.accel<0.05 and self.brake<0.05):
#             self.stuck_timer += 0.02
#             if self.stuck_timer >= STUCK_SECONDS:
#                 self.meta = 1; self.stuck_timer = 0.0
#         else:
#             self.stuck_timer = 0.0

#         # ---------- send to TORCS ----------
#         self.control.setGear(self.gear)
#         self.control.setAccel(self.accel)
#         self.control.setBrake(self.brake)
#         self.control.setSteer(self.steer)
#         self.control.setClutch(self.clutch)
#         self.control.setMeta(self.meta)

#         # ---------- CSV log ----------
#         sensors = {
#             "speedX":sim_speedX,"speedY":sim_speedY,"speedZ":sim_speedZ,
#             "angle": s.getAngle() or 0.0,
#             "trackPos": track_pos,
#             "rpm": rpm,             # live rpm
#             "gear": self.gear,      # commanded gear
#             "distRaced": s.getDistRaced() or 0.0,
#             "damage": s.getDamage() or 0.0,
#         }
#         for i,v in enumerate(s.getTrack() or [0]*19):           sensors[f"track{i}"]=v
#         for i,v in enumerate(s.getFocus() or [0]*5):            sensors[f"focus{i}"]=v
#         for i,v in enumerate(s.getWheelSpinVel() or [0]*4):     sensors[f"wheelSpinVel{i}"]=v
#         for i,v in enumerate(s.getOpponents() or [0]*36):       sensors[f"opponents{i}"]=v

#         self.logger.writerow([
#             self.track_name, self.car_model,
#             *[sensors[c] for c in ALL_NUM],
#             self.steer, self.accel, self.brake,
#             self.clutch, self.meta, self.focus_angle,
#             ml_flag,
#             int(k_left), int(k_right), int(k_throt), int(k_brake), int(k_rst)
#         ])

#         if self.meta: self.meta = 0
#         return self.control.toMsg()

#     # ───────────────────────── helpers ────────────────────────────────
#     def onShutDown(self):
#         self.logfile.close(); print("[LOG] closed")

#     def onRestart(self):
#         self.gear = self.prev_gear = 1
#         self.steer = self.accel = self.brake = 0.0
#         self.stuck_timer = 0.0
#         print("[Driver] restart state reset")

#     def _make_log_filename(self):
#         slug = lambda x: re.sub(r"[^A-Za-z0-9]", "", x)
#         base = "driving_data"
#         n = len(glob.glob(f"{base}_*.csv")) + 1
#         return f"{base}_{n:02}.csv"





"""
driver.py — TORCS Python client
• keyboard controls (W S A D, arrow ↑/↓ for gear, R for emergency reverse)
• toggle manual drive (T) and manual gear mode (G)
• gear logic: single-press + rpm-based auto-shift + clutch animation
• reverse / forward sanity, stuck detection / auto-unstuck
• CSV logger with commanded gear + live rpm
"""

import msgParser, carState, carControl
import csv, os, keyboard, threading, time, random, math

# ──────────────────────────────────────────────────────────────────────
class Driver(object):
    def __init__(self, stage):
        # ---------- TORCS plumbing ----------
        self.stage   = stage
        self.parser  = msgParser.MsgParser()
        self.state   = carState.CarState()
        self.control = carControl.CarControl()

        # ---------- drive helpers ----------
        self.steer_lock   = 0.785398
        self.max_speed    = 100

        # ----- gear & clutch -----
        self.gear              = 1
        self.prev_gear         = 1
        self.last_shift_time   = 0.0
        self.last_gear_up      = False
        self.last_gear_down    = False
        self.clutch_engaged    = 0.0
        self.clutch_duration   = 0.0
        self.shift_in_progress = False
        self.shift_start_time  = 0.0
        self.shift_type        = ""   # "up" / "down"

        # ----- race-start boost -----
        self.initial_acceleration = True
        self.initial_accel_time   = 0.0

        # ----- keyboard state -----
        self.manual_control      = True
        self.manual_gear_control = False
        self.requested_gear      = 1
        self.key_accel = self.key_brake = self.key_steer = 0.0
        self.reverse_gear  = False
        self.reverse_power = 0.0
        self.stuck_timer   = 0

        # ---------- CSV ----------
        hdr_base = [
            'Timestamp','Angle','CurrentLapTime','Damage',
            'DistanceFromStart','DistanceRaced','Fuel','Gear','LastLapTime',
            'RacePosition','RPM','SpeedX','SpeedY','SpeedZ','TrackPosition',
            'Z','WheelSpinVel1','WheelSpinVel2','WheelSpinVel3','WheelSpinVel4'
        ]
        track_hdr = [f'Track{i+1}' for i in range(19)]
        opp_hdr   = [f'OpponentX{i+1}' for i in range(36)]
        extra_hdr = ['Acceleration','Steering','Brake','Clutch','ShiftType']
        self.csv_header = hdr_base + track_hdr + opp_hdr + extra_hdr
        self.csv_filename = f"torcs_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.csv_filename,'w',newline='') as f:
            csv.writer(f).writerow(self.csv_header)

        # ---------- keyboard listener ----------
        threading.Thread(target=self.keyboard_listener,
                         daemon=True).start()

    # ───────────────────────── keyboard thread ────────────────────────
    def keyboard_listener(self):
        arrow_up_pressed   = False
        arrow_down_pressed = False
        g_key_pressed      = False
        self.key_accel = 1.0   # full throttle pre-start

        while True:
            # --- throttle / brake / reverse ---
            speed_val = getattr(self.state,'speedX',0.0) or 0.0
            if keyboard.is_pressed('w'):
                self.key_accel = 1.0
                self.key_brake = 0.0
                self.reverse_gear  = False
                self.reverse_power = 0.0
            elif keyboard.is_pressed('s'):
                if abs(speed_val) < 3.0:
                    self.reverse_gear = True
                    self.reverse_power = min(1.0, self.reverse_power+0.15)
                    self.key_brake = 0.0
                    self.key_accel = 0.0
                else:
                    self.key_brake = min(1.0, self.key_brake+0.15)
                    self.key_accel = 0.0
            else:
                self.key_accel = 0.8
                self.key_brake = max(0.0, self.key_brake-0.05)
                self.reverse_power = max(0.0, self.reverse_power-0.05)
                if abs(speed_val)<0.5 and self.reverse_power==0:
                    self.reverse_gear=False

            # --- steering (D=left, A=right) ---
            if keyboard.is_pressed('d'):
                self.key_steer = max(-1.0, self.key_steer-0.1)
            elif keyboard.is_pressed('a'):
                self.key_steer = min( 1.0, self.key_steer+0.1)
            else:
                self.key_steer *= 0.8

            # --- toggles ---
            if keyboard.is_pressed('t'):
                self.manual_control = not self.manual_control
                print(f"Manual control {'ON' if self.manual_control else 'OFF'}")
                time.sleep(0.5)
            if keyboard.is_pressed('g'):
                if not g_key_pressed:
                    self.manual_gear_control = not self.manual_gear_control
                    print(f"Manual gear {'ON' if self.manual_gear_control else 'OFF'}")
                    g_key_pressed=True
            else:
                g_key_pressed=False

            # --- arrow-key requested gear (when manual-gear mode) ---
            if self.manual_gear_control:
                cur = getattr(self.state,'gear',1)
                if keyboard.is_pressed('up'):
                    if not arrow_up_pressed and cur<6:
                        self.requested_gear = cur+1
                        print(f"Shift UP → {self.requested_gear}")
                    arrow_up_pressed=True
                else: arrow_up_pressed=False
                if keyboard.is_pressed('down'):
                    if not arrow_down_pressed and cur>1:
                        self.requested_gear = cur-1
                        print(f"Shift DOWN → {self.requested_gear}")
                    arrow_down_pressed=True
                else: arrow_down_pressed=False

            # --- emergency reverse ---
            if keyboard.is_pressed('r'):
                self.reverse_gear=True; self.reverse_power=1.0
                self.key_brake=0.0; self.key_accel=0.0
                print("!!! EMERGENCY REVERSE !!!")
                time.sleep(0.2)

            time.sleep(0.05)

    # ───────────────────── range-finder angles ────────────────────────
    def init(self):
        ang=[0]*19
        for i in range(5):  ang[i]=-90+i*15; ang[18-i]= 90-i*15
        for i in range(5,9):ang[i]=-20+(i-5)*5; ang[18-i]= 20-(i-5)*5
        return self.parser.stringify({'init':ang})

    # ─────────────────────────── drive loop ───────────────────────────
    def drive(self,msg):
        self.state.setFromMsg(msg)
        now=time.time()

        # ---------- race-start boost ----------
        if self.initial_acceleration:
            if self.initial_accel_time==0: self.initial_accel_time=now
            if now-self.initial_accel_time<5.0:
                self.control.setAccel(1.0)
                self.control.setBrake(0.0)
                if self.state.rpm>8000 and self.state.gear<6:
                    self.control.setGear(self.state.gear+1)
                    self.start_gear_shift("up")
            else:
                self.initial_acceleration=False
                print("*** BOOST OFF ***")

        # ---------- manual / auto ----------
        if self.manual_control:
            self.manual_controls()
        else:
            self.auto_controls()

        # ---------- gear / clutch ----------
        self.update_gear_logic(now)
        self.update_clutch_simulation()

        # ---------- auto-unstuck ----------
        self.auto_unstuck()

        # ---------- log ----------
        self.log_state_to_csv()
        return self.control.toMsg()

    # ───────────────── manual control block ───────────────────────────
    def manual_controls(self):
        if self.reverse_gear:
            self.control.setGear(-1)
            self.control.setAccel(self.reverse_power)
            self.control.setBrake(0.0)
        else:
            self.control.setAccel(min(1.0, self.key_accel*1.2))
            self.control.setBrake(self.key_brake)
        self.control.setSteer(self.key_steer if not self.reverse_gear
                              else self.key_steer*0.7)

        # manual gear request
        if self.manual_gear_control:
            cur = self.state.getGear()
            if cur!=self.requested_gear and 1<=self.requested_gear<=6:
                self.control.setGear(self.requested_gear)
                self.start_gear_shift("up" if self.requested_gear>cur else "down")

    # ───────────────────── auto control stub ──────────────────────────
    def auto_controls(self):
        steer = (self.state.angle - self.state.trackPos*0.5)/self.steer_lock
        self.control.setSteer(max(-1,min(1,steer)))
        self.control.setAccel(1.0 if self.state.speedX<self.max_speed else 0.0)
        self.control.setBrake(0.0)

    # ───────────────────── gear logic core ────────────────────────────
    def update_gear_logic(self, now):
        rpm   = self.state.getRpm()   or 0.0
        speed = getattr(self.state,'speedX',0.0) or 0.0
        gear  = self.control.getGear() or self.gear

        # manual arrow-key shifts (outside manual-gear mode)
        if not self.manual_gear_control:
            if keyboard.is_pressed('up'):
                if not self.last_gear_up:
                    gear=min(gear+1,7); self.last_shift_time=now
                    self.last_gear_up=True
            else: self.last_gear_up=False
            if keyboard.is_pressed('down'):
                if not self.last_gear_down:
                    gear=max(gear-1,1); self.last_shift_time=now
                    self.last_gear_down=True
            else: self.last_gear_down=False

        # auto shift (cool-down 0.5 s)
        if now-self.last_shift_time>0.5 and not self.manual_gear_control:
            if rpm>8000 and gear<7:
                gear+=1; self.last_shift_time=now
            elif rpm<4000 and gear>1:
                gear-=1; self.last_shift_time=now

        # direction sanity
        if speed<0 and gear>0: gear=-1
        elif speed>0 and gear<0: gear=1

        gear=max(min(gear,7),-1)
        if gear==0: gear=1

        if gear!=self.prev_gear:
            self.start_gear_shift("up" if gear>self.prev_gear else "down")
        self.prev_gear=gear
        self.gear=gear
        self.control.setGear(gear)

    # ───────────────── clutch simulation ──────────────────────────────
    def start_gear_shift(self,shift_type):
        if self.shift_in_progress: return
        self.shift_in_progress=True
        self.shift_start_time=time.time()
        self.shift_type=shift_type
        self.clutch_duration=0.3+random.random()*0.3
        self.clutch_engaged=0.0

    def update_clutch_simulation(self):
        if not self.shift_in_progress: return
        t=time.time()-self.shift_start_time
        if t<self.clutch_duration:
            prog=t/self.clutch_duration
            self.clutch_engaged= prog*2 if prog<0.5 else 2.0-prog*2
        else:
            self.shift_in_progress=False
            self.clutch_engaged=0.0
            self.shift_type=""

    # ───────────────── auto-unstuck ───────────────────────────────────
    def auto_unstuck(self):
        sx = getattr(self.state,'speedX',0.0) or 0.0
        if abs(sx)<0.5 and self.key_accel>0.5:
            self.stuck_timer+=1
            if self.stuck_timer>40:
                print("Auto-unstuck: reverse")
                self.reverse_gear=True; self.reverse_power=1.0
                self.control.setGear(-1)
                self.control.setAccel(1.0)
                self.control.setBrake(0.0)
                self.stuck_timer=0
        else:
            self.stuck_timer=0

    # ───────────────────────── CSV logger ─────────────────────────────
    def log_state_to_csv(self):
        s=self.state
        row=[
            time.time(),
            s.angle,s.curLapTime,s.damage,
            s.distFromStart,s.distRaced,s.fuel,
            self.control.getGear(),                   # commanded gear
            s.lastLapTime,s.racePos,
            s.rpm,
            s.speedX,s.speedY,s.speedZ,
            s.trackPos,s.z,
            *(s.wheelSpinVel[:4] if hasattr(s,'wheelSpinVel') else [0]*4),
            *(s.track[:19]         if hasattr(s,'track') else [0]*19),
            *(s.opponents[:36]     if hasattr(s,'opponents') else [0]*36),
            self.control.getAccel(), self.control.getSteer(),
            self.control.getBrake(), self.clutch_engaged,
            1 if self.shift_type=="up" else 0 if self.shift_type=="down" else ""
        ]
        with open(self.csv_filename,'a',newline='') as f:
            csv.writer(f).writerow(row)

    # ─────────────────── TORCS callbacks ──────────────────────────────
    def onShutDown(self):
        print(f"Driver shut down – log saved to {self.csv_filename}")

    def onRestart(self):
        self.csv_filename=f"torcs_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.csv_filename,'w',newline='') as f:
            csv.writer(f).writerow(self.csv_header)
        self.gear=self.prev_gear=1
        self.initial_acceleration=True
        print(f"Driver restarted – new log {self.csv_filename}")
