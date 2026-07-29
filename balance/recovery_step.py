"""
balance/recovery_step.py

Stepping strategy: the last line of defence.  When the capture point leaves the
support polygon, neither ankle nor hip torque can stop the fall -- the robot
must take a step *toward the capture point* to enlarge its base of support and
catch itself (Pratt et al., 2006).

The planner produces **one step at a time**, each sized to the *current*
capture point.  The controller re-invokes it after every step until the CoM has
slowed down, so the support region chases the (receding) capture point for as
many steps as the disturbance requires -- a light push is caught in one step, a
hard shove in several.

Foot positions are (x_forward, y_left, z_fold) relative to the hip, in mm;
larger z means the leg is more folded (foot lifts toward the hip).
"""
import numpy as np


class RecoveryStepPlanner:
    def __init__(self, sit=50.0, lift=28.0, step_gain=1.1,
                 min_step=45.0, max_step=190.0, max_back_step=60.0,
                 step_duration=0.16):
        self.sit = sit
        self.lift = lift
        self.step_gain = step_gain
        self.min_step = min_step
        self.max_step = max_step
        self.max_back_step = max_back_step
        self.step_duration = step_duration

    def lead_foot(self, offset_xy):
        """Which foot swings first: step out on the side you fall toward."""
        ox, oy = offset_xy
        if abs(oy) > abs(ox):
            return "L" if oy > 0 else "R"
        return "R"

    def _step_vector(self, cp_offset_xy):
        """Foot placement (Lx, Ly) in mm toward the capture point."""
        ox, oy = cp_offset_xy
        norm = np.hypot(ox, oy)
        mag = float(np.clip(norm * 1000.0 * self.step_gain,
                            self.min_step, self.max_step))
        if norm < 1e-9:
            return self.min_step, 0.0
        Lx, Ly = mag * ox / norm, mag * oy / norm
        if Lx < 0:
            Lx = max(Lx, -self.max_back_step)
        return Lx, Ly

    def plan_single(self, cp_offset_xy, lead, foot_r, foot_l):
        """
        Plan ONE recovery step toward the current capture point.

        cp_offset_xy : capture-point offset (xi - CoM) in metres.
        lead         : "R" or "L", the foot to swing this step.
        foot_r/foot_l: current foot positions (mm, hip frame) to start from.
        Returns (waypoints, new_foot_r, new_foot_l, next_lead) where waypoints is
        a list of (right_point, left_point, duration).
        """
        Lx, Ly = self._step_vector(cp_offset_xy)
        sit, lift, dur = self.sit, self.lift, self.step_duration
        foot_r = np.asarray(foot_r, float)
        foot_l = np.asarray(foot_l, float)

        stance = np.array([0.0, 0.0, sit])
        if lead == "R":
            ly = min(Ly, 0.0)
            swing_hi = np.array([Lx, ly, sit + lift])
            swing_lo = np.array([Lx, ly, sit])
            waypoints = [(swing_hi, foot_l.copy(), dur * 0.5),
                         (swing_lo, stance.copy(), dur * 0.5)]
            return waypoints, swing_lo, stance, "L"
        else:
            ly = max(Ly, 0.0)
            swing_hi = np.array([Lx, ly, sit + lift])
            swing_lo = np.array([Lx, ly, sit])
            waypoints = [(foot_r.copy(), swing_hi, dur * 0.5),
                         (stance.copy(), swing_lo, dur * 0.5)]
            return waypoints, stance, swing_lo, "R"

    def decide_n_steps(self, speed):
        """How many steps to commit to, based on the disturbance speed."""
        return int(np.clip(round(speed / 0.25) + 1, 2, 4))

    def plan_full(self, cp_offset_xy, speed, n_steps=None,
                  include_recenter=False):
        """Plan a COMPLETE, committed recovery: ``n_steps`` protective steps
        toward the capture point (feet alternating) followed by a recentre.

        Committing to the whole maneuver -- rather than re-deciding after every
        step -- avoids the limit-cycle thrashing that a per-step controller
        suffers when the residual motion keeps re-triggering it.
        Returns a flat list of (right_point, left_point, duration) way-points.
        """
        if n_steps is None:
            n_steps = self.decide_n_steps(speed)
        lead = self.lead_foot(cp_offset_xy)
        foot_r = np.array([0.0, 0.0, self.sit])
        foot_l = np.array([0.0, 0.0, self.sit])
        waypoints = []
        for _ in range(n_steps):
            wps, foot_r, foot_l, lead = self.plan_single(
                cp_offset_xy, lead, foot_r, foot_l)
            waypoints.extend(wps)
        if include_recenter:
            waypoints.extend(self.recenter())
        return waypoints

    def recenter(self, duration=0.8):
        """Way-point that brings both feet back under the hips.

        Deliberately slow, and deliberately NOT part of the catch: closing the
        stance is only safe once the CoM has actually settled.  Doing it
        immediately after the catch step removes the very support base the step
        just created, and the robot tips over sideways.
        """
        c = np.array([0.0, 0.0, self.sit])
        return [(c.copy(), c.copy(), duration)]
