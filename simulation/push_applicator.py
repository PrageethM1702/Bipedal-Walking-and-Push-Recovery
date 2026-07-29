"""
simulation/push_applicator.py

Applies disturbance forces ("pushes") to the robot's trunk so the balance
controller can be exercised.  A push is a force of a given magnitude and
direction held for a short duration (an impulse).  Pushes can be:

  * triggered manually            -> ``push(direction, magnitude)``
  * scheduled at fixed sim times  -> ``schedule(time, direction, magnitude)``
  * generated at random intervals -> ``enable_random(...)``

The applicator is ticked once per simulation step; it (re)applies the active
push force and returns a description of any push that fired this tick, which the
demo/harness logs.
"""
import numpy as np

DIRECTIONS = {
    "forward":  np.array([1.0, 0.0, 0.0]),
    "backward": np.array([-1.0, 0.0, 0.0]),
    "left":     np.array([0.0, 1.0, 0.0]),
    "right":    np.array([0.0, -1.0, 0.0]),
}


class PushApplicator:
    def __init__(self, robot, dt, duration=0.05, link_index=-1, rng_seed=0):
        self.robot = robot
        self.dt = dt
        self.duration = duration
        self.link_index = link_index
        self.t = 0.0

        self._active_ticks = 0
        self._active_force = np.zeros(3)
        self._last_event = None

        self._schedule = []
        self._random = None
        self._rng = np.random.default_rng(rng_seed)
        self._next_random = None

    def schedule(self, time, direction, magnitude):
        vec = self._resolve(direction) * magnitude
        label = f"{self._name(direction)} {magnitude:.0f}N"
        self._schedule.append((time, vec, label))
        self._schedule.sort(key=lambda e: e[0])

    def enable_random(self, interval=(3.0, 6.0), magnitude=(20.0, 80.0),
                      directions=("forward", "backward", "left", "right")):
        self._random = dict(interval=interval, magnitude=magnitude,
                            directions=list(directions))
        self._next_random = self.t + self._rng.uniform(*interval)

    def push(self, direction, magnitude):
        """Trigger a push right now."""
        self._active_force = self._resolve(direction) * magnitude
        self._active_ticks = max(1, int(self.duration / self.dt))
        self._last_event = dict(time=self.t, force=self._active_force.copy(),
                                label=f"{self._name(direction)} {magnitude:.0f}N")
        return self._last_event

    def update(self):
        """Call once per simulation tick (before stepSimulation).
        Returns a push-event dict if a push *starts* this tick, else None."""
        event = None

        while self._schedule and self._schedule[0][0] <= self.t:
            _, vec, label = self._schedule.pop(0)
            self._active_force = vec.copy()
            self._active_ticks = max(1, int(self.duration / self.dt))
            event = dict(time=self.t, force=vec.copy(), label=label)

        if self._random is not None and self.t >= self._next_random:
            d = self._rng.choice(self._random["directions"])
            m = self._rng.uniform(*self._random["magnitude"])
            self._active_force = self._resolve(d) * m
            self._active_ticks = max(1, int(self.duration / self.dt))
            event = dict(time=self.t, force=self._active_force.copy(),
                         label=f"{d} {m:.0f}N")
            self._next_random = self.t + self._rng.uniform(*self._random["interval"])

        if self._active_ticks > 0:
            self.robot.apply_external_force(self._active_force,
                                            link_index=self.link_index)
            self._active_ticks -= 1

        if event is not None:
            self._last_event = event
        self.t += self.dt
        return event

    def _resolve(self, direction):
        if isinstance(direction, str):
            return DIRECTIONS[direction].copy()
        v = np.asarray(direction, dtype=float)
        if v.shape[0] == 2:
            v = np.array([v[0], v[1], 0.0])
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def _name(self, direction):
        return direction if isinstance(direction, str) else "custom"
