"""
tools/data_logger.py

Records a run of the simulation to ``logs/<run-name>/`` for later analysis:

  * ``timeseries.csv`` -- one row per logged tick (CoM, capture point, support
    margin, trunk tilt, controller state, foot positions/contacts),
  * ``events.csv``     -- push events and state transitions,
  * ``footsteps.csv``  -- planned vs. actually measured foot touch-downs,
  * ``summary.json``   -- headline metrics for the run.

Actual footsteps are detected as the rising edge of each foot's ground contact;
planned footsteps come from the recovery planner (``controller.planned_steps``).
"""
import csv
import json
import os
import time

import numpy as np

DEFAULT_LOG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


class DataLogger:
    def __init__(self, robot, controller, dt, run_name=None,
                 log_root=DEFAULT_LOG_ROOT, decimate=10):
        self.robot = robot
        self.controller = controller
        self.dt = dt
        self.decimate = max(1, int(decimate))
        self.run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
        self.dir = os.path.join(log_root, self.run_name)
        os.makedirs(self.dir, exist_ok=True)

        self.rows = []
        self.events = []
        self.actual_steps = []

        self._i = 0
        self._prev_contacts = (True, True)
        self._prev_state = None
        self._airborne = {"left": False, "right": False}
        self._last_step_tick = {"left": -10 ** 9, "right": -10 ** 9}
        self.foot_rest_z = 0.030
        self.lift_clear = self.foot_rest_z + 0.015
        self.plant_z = self.foot_rest_z + 0.006
        self.min_step_gap = int(0.10 / dt)

    def log(self, sense, push_event=None):
        """Call once per simulation tick, after ``controller.step()``."""
        t = self._i * self.dt
        lc, rc = self.robot.get_foot_contacts()
        lf = self.robot.get_left_foot_pos()
        rf = self.robot.get_right_foot_pos()
        state = sense["state"].value if hasattr(sense["state"], "value") \
            else str(sense["state"])

        if push_event is not None:
            f = np.asarray(push_event["force"], dtype=float)
            com = sense["com"]
            self.events.append(dict(t=round(t, 4), kind="push",
                                    detail=push_event["label"],
                                    fx=float(f[0]), fy=float(f[1]),
                                    com_x=float(com[0]), com_y=float(com[1])))
        if state != self._prev_state:
            self.events.append(dict(t=round(t, 4), kind="state",
                                    detail=state))
            self._prev_state = state

        for side, now, pos in (("left", lc, lf), ("right", rc, rf)):
            if float(pos[2]) > self.lift_clear:
                self._airborne[side] = True
            elif float(pos[2]) < self.plant_z and self._airborne[side] and \
                    (self._i - self._last_step_tick[side]) > self.min_step_gap:
                self.actual_steps.append(dict(t=round(t, 4), side=side,
                                              x=float(pos[0]), y=float(pos[1])))
                self._airborne[side] = False
                self._last_step_tick[side] = self._i
        self._prev_contacts = (lc, rc)

        if self._i % self.decimate == 0:
            com = sense["com"]; vel = sense["vel"]; cp = sense["cp"]
            rpy = sense["rpy"]
            self.rows.append(dict(
                t=round(t, 4),
                com_x=float(com[0]), com_y=float(com[1]), com_z=float(com[2]),
                vel_x=float(vel[0]), vel_y=float(vel[1]),
                speed=float(sense["speed"]),
                cp_x=float(cp[0]), cp_y=float(cp[1]),
                cp_margin=float(sense["margin"]),
                roll=float(rpy[0]), pitch=float(rpy[1]), yaw=float(rpy[2]),
                tilt=float(sense["tilt"]),
                lf_x=float(lf[0]), lf_y=float(lf[1]), lf_z=float(lf[2]),
                rf_x=float(rf[0]), rf_y=float(rf[1]), rf_z=float(rf[2]),
                left_contact=int(lc), right_contact=int(rc),
                state=state,
            ))
        self._i += 1

    def save(self, extra_summary=None):
        self._write_csv("timeseries.csv", self.rows)
        self._write_csv("events.csv", self.events)

        planned = list(getattr(self.controller, "planned_steps", []))
        steps = [dict(kind="planned", **s) for s in planned] + \
                [dict(kind="actual", **s) for s in self.actual_steps]
        self._write_csv("footsteps.csv", steps)

        duration = self._i * self.dt
        com = self.robot.get_com_position()
        summary = dict(
            run=self.run_name,
            duration_s=round(duration, 3),
            ticks=self._i,
            final_com=[float(com[0]), float(com[1]), float(com[2])],
            fell=bool(com[2] < 0.13),
            n_pushes=sum(1 for e in self.events if e["kind"] == "push"),
            n_planned_steps=len(planned),
            n_actual_steps=len(self.actual_steps),
            states_visited=sorted({e["detail"] for e in self.events
                                   if e["kind"] == "state"}),
        )
        if extra_summary:
            summary.update(extra_summary)
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return self.dir, summary

    def _write_csv(self, name, rows):
        path = os.path.join(self.dir, name)
        if not rows:
            open(path, "w").close()
            return
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)
