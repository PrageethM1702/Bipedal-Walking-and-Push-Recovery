"""
control/main_controller.py

Ties the whole system together into a single per-tick controller:

    sense  ->  capture point + support polygon  ->  state machine  ->  action

Actions:
  * STAND / WALK          : play the nominal pose / gait, stabilised by a gentle
                            continuous ankle strategy.
  * ANKLE_RECOVER         : stronger ankle strategy.
  * HIP_RECOVER           : ankle + hip strategy (trunk flail).
  * STEP_RECOVER          : execute a capture-point step plan through the leg IK,
                            overriding the gait until the robot is settled.

The controller emits one 12-vector of joint targets per call to ``step`` and is
agnostic to how the simulation is stepped, so the same controller drives the
GUI demo and the headless evaluation harness.
"""
import numpy as np

from kinematics.inverse_kinematics import solve_both_legs, PELVIC_INTERVAL
from balance.capture_point import capture_point, omega
from balance.support_polygon import SupportPolygon
from balance.ankle_strategy import AnkleStrategy
from balance.hip_strategy import HipStrategy
from balance.recovery_step import RecoveryStepPlanner
from control.state_machine import BalanceStateMachine, State

HALF_PELVIS = PELVIC_INTERVAL / 2000.0


class MainController:
    def __init__(self, robot, walk_generator, dt,
                 sit=50.0, walk_action_time=0.06,
                 enable_walk=False):
        self.robot = robot
        self.dt = dt
        self.sit = sit
        self.enable_walk = enable_walk

        self.stand_pose = solve_both_legs((0, 0, sit), (0, 0, sit))

        self.ankle = AnkleStrategy(kp_pitch=1.1, kd_pitch=0.04,
                                   kp_roll=1.1, kd_roll=0.04, max_offset=0.45)
        self.hip = HipStrategy(k_pitch=0.25, k_roll=0.25)
        self.stepper = RecoveryStepPlanner(sit=sit, step_gain=1.4,
                                           step_duration=0.12)
        self.fsm = BalanceStateMachine()

        self.walk = walk_generator
        self._walk_intro = None
        self._walk_entry = self.stand_pose.copy()
        self._walk_stream = self._build_walk_stream(walk_action_time)
        self._walk_idx = 0
        self._intro_idx = 0
        self._entry_idx = 0
        self._entry_ticks = max(1, int(0.6 / dt))
        self._intro_done = False

        self._recover_queue = []
        self._foot_r = np.array([0.0, 0.0, sit])
        self._foot_l = np.array([0.0, 0.0, sit])
        self._lead = "R"
        self._recentering = False
        self._rec_phase = None
        self._hold_ticks = 0
        self._settled_ticks = 0
        self.settle_speed = 0.14
        self.settle_tilt = 0.22
        self._settled_needed = int(0.30 / dt)
        self._hold_max = int(2.5 / dt)
        self._cooldown = 0
        self._cooldown_ticks = int(0.35 / dt)
        self._t = 0
        self._arm_ticks = int(0.8 / dt)

        self._prev_rpy = self.robot.get_base_orientation()
        self._cur = self.stand_pose.copy()
        self.state = State.STAND

        self.planned_steps = []
        self._walk_swing = {"left": False, "right": False}
        self._swing_clear = 0.045

    def _interp_frames(self, frames, start, action_time):
        """Linearly interpolate through ``frames`` at one row per tick."""
        n_sub = max(1, int(action_time / self.dt))
        stream = []
        prev = np.array(start, dtype=float)
        for f in frames:
            f = np.array(f, dtype=float)
            for s in range(n_sub):
                a = (s + 1) / n_sub
                stream.append(prev + a * (f - prev))
            prev = f
        return stream, prev

    def _build_walk_stream(self, action_time):
        """Expand the gait keyframes into per-tick angle streams.

        The walk must begin with the generator's *start* motion, which builds up
        the lateral sway rhythm; dropping straight into the cyclic frames leaves
        the body sway out of phase with the swing leg and the robot topples
        after a few strides.  We therefore keep a one-shot lead-in stream and a
        separate cyclic stream that loops afterwards.
        """
        if self.walk is None or not hasattr(self.walk, "walkAnglesWalkingRight"):
            self._walk_intro = None
            return None

        start_pose = self.walk.walkAnglesStartLeft[0]
        intro, last = self._interp_frames(
            list(self.walk.walkAnglesStartLeft), start_pose, action_time)
        cyc_frames = list(self.walk.walkAnglesWalkingRight) + \
            list(self.walk.walkAnglesWalkingLeft)
        cycle, _ = self._interp_frames(cyc_frames, last, action_time)

        self._walk_intro = np.array(intro)
        self._walk_entry = np.array(start_pose, dtype=float)
        return np.array(cycle)

    def _expand_step_plan(self, waypoints, start_r, start_l):
        """Turn foot-point way-points into a per-tick joint-target queue,
        interpolating from the current foot positions."""
        queue = []
        rp = np.asarray(start_r, float).copy()
        lp = np.asarray(start_l, float).copy()
        for r1, l1, dur in waypoints:
            r1 = np.asarray(r1, float); l1 = np.asarray(l1, float)
            n = max(1, int(dur / self.dt))
            r0, l0 = rp.copy(), lp.copy()
            for s in range(n):
                a = (s + 1) / n
                queue.append(solve_both_legs(tuple(r0 + a * (r1 - r0)),
                                             tuple(l0 + a * (l1 - l0))))
            rp, lp = r1, l1
        return queue, rp, lp

    def start_walking(self):
        """Begin walking: ease into the gait's start motion, then cycle."""
        self.enable_walk = True
        self.fsm.state = State.WALK
        self._walk_idx = 0
        self._intro_idx = 0
        self._entry_idx = 0
        self._intro_done = False

    def stop_walking(self):
        self.enable_walk = False
        self.fsm.state = State.STAND

    def sense(self):
        com = self.robot.get_com_position()
        vel = self.robot.get_com_velocity()
        rpy = self.robot.get_base_orientation()
        rate = self.robot.get_base_velocity()[1]
        self._prev_rpy = rpy
        z = max(com[2], 1e-3)
        cp = capture_point(com[:2], vel[:2], z)
        cp_offset = vel[:2] / omega(z)
        sp = SupportPolygon.from_robot(self.robot)
        margin = sp.margin(cp)
        speed = float(np.hypot(vel[0], vel[1]))
        tilt = max(abs(rpy[0]), abs(rpy[1]))
        return dict(com=com, vel=vel, rpy=rpy, rate=rate, cp=cp,
                    cp_offset=cp_offset, margin=margin, speed=speed, tilt=tilt)

    def step(self):
        """Compute and apply one tick of joint targets. Returns telemetry."""
        self._t += 1
        s = self.sense()
        step_active = bool(self._recover_queue) or self._rec_phase is not None

        if self._t < self._arm_ticks and not step_active:
            return self._emit_nominal(s, State.STAND)

        if self._cooldown > 0 and not step_active:
            self._cooldown -= 1
            return self._emit_nominal(
                s, State.WALK if self.enable_walk else State.STAND)

        self.state = self.fsm.update(
            s["margin"], s["speed"], s["tilt"], com_z=s["com"][2],
            walking=self.enable_walk, step_active=step_active)

        if self.state == State.FALLEN:
            angles = self._cur
            self._cur = angles
            self.robot.apply_motor_angles(angles)
            s["state"] = self.state
            return s

        if self.state == State.STEP_RECOVER:
            if self._rec_phase is None:
                wps = self.stepper.plan_full(s["cp_offset"], s["speed"],
                                             include_recenter=False)
                self._record_planned_steps(wps, s)
                self._recover_queue, self._foot_r, self._foot_l = \
                    self._expand_step_plan(wps, self._foot_r, self._foot_l)
                self._rec_phase = "catch"
                self._hold_ticks = 0
                self._settled_ticks = 0

            if self._recover_queue:
                angles = self._recover_queue.pop(0)
            elif self._rec_phase == "catch":
                self._rec_phase = "hold"
                angles = solve_both_legs(tuple(self._foot_r),
                                         tuple(self._foot_l))
            elif self._rec_phase == "hold":
                self._hold_ticks += 1
                calm = (s["speed"] < self.settle_speed
                        and s["tilt"] < self.settle_tilt)
                self._settled_ticks = self._settled_ticks + 1 if calm else 0
                if (self._settled_ticks >= self._settled_needed
                        or self._hold_ticks >= self._hold_max):
                    wps = self.stepper.recenter(duration=0.8)
                    self._recover_queue, self._foot_r, self._foot_l = \
                        self._expand_step_plan(wps, self._foot_r, self._foot_l)
                    self._rec_phase = "recenter"
                angles = solve_both_legs(tuple(self._foot_r),
                                         tuple(self._foot_l))
            else:
                self._rec_phase = None
                self._recentering = True
                angles = self.stand_pose.copy()

            self.ankle.apply(angles, s["rpy"][0], s["rpy"][1],
                             s["rate"][0], s["rate"][1])
            self._cur = angles
            self.robot.apply_motor_angles(angles)
            s["state"] = self.state
            return s

        if self._recentering:
            self._recentering = False
            self._cooldown = self._cooldown_ticks
            self._foot_r = np.array([0.0, 0.0, self.sit])
            self._foot_l = np.array([0.0, 0.0, self.sit])
            self._lead = "R"
        return self._emit_nominal(s, self.state)

    def _record_walk_step(self, s):
        """Record the gait's nominal foot placements while walking.

        The gait is cyclic, so each swing foot is *planned* to land half a
        stride ahead of the trunk, on its own side of the body.  We emit one
        planned placement per swing phase (detected from the swing foot leaving
        the ground) so the walk can be compared against the measured
        touch-downs in the same plot as the recovery steps."""
        stride_m = getattr(self.walk, "_l", 90.0) / 1000.0
        lf = self.robot.get_left_foot_pos()
        rf = self.robot.get_right_foot_pos()
        com = s["com"]
        for side, foot in (("left", lf), ("right", rf)):
            lifted = float(foot[2]) > self._swing_clear
            if lifted and not self._walk_swing[side]:
                y_hip = HALF_PELVIS if side == "left" else -HALF_PELVIS
                self.planned_steps.append(dict(
                    t=self._t * self.dt, side=side,
                    x=float(com[0] + stride_m * 0.5),
                    y=float(com[1] + y_hip)))
            self._walk_swing[side] = lifted

    def _record_planned_steps(self, waypoints, s):
        """Store where the planner *intends* each foot to land, in world
        coordinates, so it can be compared against where the feet actually end
        up (see tools/plot_logs.py).  Foot points are hip-relative millimetres;
        the trunk position and yaw map them into the world."""
        com = s["com"]
        yaw = s["rpy"][2]
        c, sn = np.cos(yaw), np.sin(yaw)
        seen = set()
        for r_pt, l_pt, _dur in waypoints:
            for side, pt in (("right", r_pt), ("left", l_pt)):
                if abs(float(pt[2]) - self.sit) > 1e-6:
                    continue
                key = (side, round(float(pt[0]), 3), round(float(pt[1]), 3))
                if key in seen:
                    continue
                seen.add(key)
                x_mm, y_mm = float(pt[0]), float(pt[1])
                y_hip = (HALF_PELVIS if side == "left" else -HALF_PELVIS)
                dx, dy = x_mm / 1000.0, y_mm / 1000.0 + y_hip
                self.planned_steps.append(dict(
                    t=self._t * self.dt, side=side,
                    x=float(com[0] + c * dx - sn * dy),
                    y=float(com[1] + sn * dx + c * dy)))

    def _emit_nominal(self, s, state):
        """Command the nominal pose/gait plus continuous stabilisation."""
        self.state = state
        walking_now = (self.enable_walk and self._walk_stream is not None
                       and state in (State.WALK, State.ANKLE_RECOVER,
                                     State.HIP_RECOVER))
        if walking_now:
            self._record_walk_step(s)
        if walking_now:
            if not self._intro_done and self._walk_intro is not None:
                if self._entry_idx < self._entry_ticks:
                    a = (self._entry_idx + 1) / self._entry_ticks
                    angles = (self.stand_pose * (1 - a)
                              + self._walk_entry * a)
                    self._entry_idx += 1
                elif self._intro_idx < len(self._walk_intro):
                    angles = self._walk_intro[self._intro_idx].copy()
                    self._intro_idx += 1
                else:
                    self._intro_done = True
                    angles = self._walk_stream[0].copy()
                    self._walk_idx = 1
            else:
                angles = self._walk_stream[self._walk_idx % len(self._walk_stream)].copy()
                self._walk_idx += 1
        else:
            angles = self.stand_pose.copy()

        if not walking_now or state != State.WALK:
            self.ankle.apply(angles, s["rpy"][0], s["rpy"][1],
                             s["rate"][0], s["rate"][1])
        if state == State.HIP_RECOVER:
            self.hip.apply(angles, s["vel"][:2])

        self._cur = angles
        self.robot.apply_motor_angles(angles)
        s["state"] = state
        return s
