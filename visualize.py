"""
visualize.py -- guided GUI showcase of every capability.

Opens a PyBullet window and runs through the whole system, phase by phase,
with on-screen captions and live overlays:

    1. Standing balance          -- the robot holds itself up
    2. Ankle strategy            -- small pushes absorbed at the ankles
    3. Hip strategy              -- a faster push adds trunk motion
    4. Walking                   -- feed-forward gait
    5. Walking under disturbance -- pushed while walking

Stepping recovery is implemented (balance/recovery_step.py) but is NOT part of
the tour: this robot's 5 rad/s servos cannot swing a leg fast enough to plant a
catch step in time, so it is not a capability we claim.  See README.

Overlays drawn every frame:
    * green  sphere/cross  -- centre of mass, projected on the ground
    * red/blue cross       -- capture point (blue = inside the support polygon
                              and therefore recoverable without stepping,
                              red = outside, a step is needed)
    * yellow outline       -- support polygon (the convex hull of the feet
                              currently touching the ground)
    * magenta arrow        -- a push, drawn while the impulse is applied
    * text above the robot -- phase, controller state, and live numbers

Usage
-----
    python visualize.py                 # full guided showcase, then free play
    python visualize.py --phase walk    # jump straight to one phase
    python visualize.py --free          # skip the tour, go straight to manual
    python visualize.py --speed 0.5     # play at half speed
    python visualize.py --list          # list the phase names

Keyboard (any time)
-------------------
    arrow keys : push the robot (forward / back / left / right)
    SHIFT+arrow: hard push (past the ankle limit -- it will fall)
    SPACE      : toggle walking
    R          : reset the robot
    N          : skip to the next phase
    Q          : quit

NOTE: PyBullet's viewer reserves A C D G I J K L O P S V W, Esc, F1 and F3 for
its own debug functions (W is wireframe, S is shadows, G hides the panels), so
this demo only binds keys outside that set.
"""
import argparse
import os
import sys
import time

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.sim_manager import SimManager
from simulation.robot_interface import RobotInterface, SPAWN_Z
from simulation.push_applicator import PushApplicator
from walking.walk_generator import WalkGenerator
from control.main_controller import MainController
from balance.support_polygon import SupportPolygon

GAIT = dict(bodyMovePoint=8, legMovePoint=8, height=50, stride=90, sit=50,
            swayBody=30, swayFoot=0, bodyPositionForwardPlus=5, swayShift=3,
            liftPush=0.5, landPull=0.7, timeStep=0.06)

STATE_COLOR = {
    "STAND":         (0.55, 0.75, 1.0),
    "WALK":          (0.45, 0.95, 0.45),
    "ANKLE_RECOVER": (1.0, 0.72, 0.2),
    "HIP_RECOVER":   (0.85, 0.5, 0.95),
    "STEP_RECOVER":  (1.0, 0.35, 0.3),
    "FALLEN":        (0.6, 0.6, 0.6),
}

PHASES = [
    dict(name="intro", secs=4.0, walk=False, pushes=[],
         title="1. STANDING BALANCE",
         blurb="Holding the standing pose. Watch the capture point (cross)"
               " stay inside the yellow support polygon."),
    dict(name="ankle", secs=13.0, walk=False,
         pushes=[(1.5, "forward", 15), (4.5, "backward", 15),
                 (7.5, "left", 18), (10.5, "right", 18)],
         title="2. ANKLE STRATEGY  (small pushes)",
         blurb="The capture point stays inside the polygon, so the ankles"
               " alone bring the robot back upright. No stepping."),
    dict(name="hip", secs=8.0, walk=False,
         pushes=[(1.5, "forward", 22), (5.0, "left", 25)],
         title="3. ANKLE + HIP STRATEGY  (medium pushes)",
         blurb="Faster disturbance: the trunk is driven against the fall"
               " on top of the ankle correction."),
    dict(name="walk", secs=11.0, walk=True, pushes=[],
         title="4. WALKING",
         blurb="Feed-forward gait: sinusoidal swing feet plus the lateral"
               " body sway that shifts weight onto the stance foot."),
    dict(name="walkpush", secs=18.0, walk=True,
         pushes=[(3.4, "left", 12), (7.4, "right", 12),
                 (11.4, "backward", 12), (15.4, "left", 12)],
         title="5. WALKING UNDER DISTURBANCE",
         blurb="Pushed mid-stride and keeps walking. Balance feedback is"
               " gated so it does not fight the gait's own sway."),
]


class Pacer:
    """Throttles the 2 kHz simulation toward wall-clock speed without
    sleeping on every single tick (which would dominate the runtime)."""

    def __init__(self, dt, speed=1.0, chunk=25):
        self.dt = dt
        self.speed = max(speed, 1e-3)
        self.chunk = chunk
        self._n = 0
        self._t0 = time.perf_counter()
        self._sim_t = 0.0

    def tick(self):
        self._n += 1
        self._sim_t += self.dt
        if self._n % self.chunk:
            return
        target = self._sim_t / self.speed
        lag = target - (time.perf_counter() - self._t0)
        if lag > 0:
            time.sleep(lag)


class Overlay:
    """All the debug geometry and text drawn into the GUI."""

    def __init__(self, gui=True):
        self.gui = gui
        self.ids = {}
        self.push_arrow = None
        self.push_until = 0.0

    def _line(self, key, a, b, colour, width=2.0):
        if not self.gui:
            return
        kw = dict(lineColorRGB=colour, lineWidth=width)
        if key in self.ids:
            kw["replaceItemUniqueId"] = self.ids[key]
        self.ids[key] = p.addUserDebugLine(a, b, **kw)

    def _text(self, key, msg, pos, colour, size=1.3):
        if not self.gui:
            return
        kw = dict(textColorRGB=colour, textSize=size)
        if key in self.ids:
            kw["replaceItemUniqueId"] = self.ids[key]
        self.ids[key] = p.addUserDebugText(msg, pos, **kw)

    def _cross(self, key, xy, colour, r=0.035, z=0.004):
        x, y = float(xy[0]), float(xy[1])
        self._line(key + "_a", [x - r, y, z], [x + r, y, z], colour, 3.0)
        self._line(key + "_b", [x, y - r, z], [x, y + r, z], colour, 3.0)

    def draw(self, robot, sense, phase, elapsed, total, walking, sim_t):
        if not self.gui:
            return
        com = sense["com"]
        state = sense["state"].value
        colour = STATE_COLOR.get(state, (1, 1, 1))

        sp = SupportPolygon.from_robot(robot)
        v = sp.vertices
        for i in range(len(v)):
            a = v[i]
            b = v[(i + 1) % len(v)]
            self._line(f"sp{i}", [a[0], a[1], 0.003], [b[0], b[1], 0.003],
                       (1.0, 0.85, 0.1), 2.5)

        self._cross("com", com[:2], (0.2, 0.95, 0.35))
        cp_in = sense["margin"] >= 0.0
        self._cross("cp", sense["cp"], (0.2, 0.6, 1.0) if cp_in else (1.0, 0.2, 0.2))

        base = [com[0], com[1], 0.62]
        self._text("title", phase["title"], base, (1, 1, 1), 1.5)
        self._text("blurb", phase["blurb"],
                   [base[0], base[1], base[2] - 0.075], (0.8, 0.85, 0.9), 1.0)
        self._text("state",
                   f"state: {state}    {'WALKING' if walking else 'STANDING'}",
                   [base[0], base[1], base[2] - 0.145], colour, 1.25)
        self._text("nums",
                   f"CP margin {sense['margin']:+.3f} m   "
                   f"speed {sense['speed']:.2f} m/s   "
                   f"tilt {sense['tilt']:.2f} rad   "
                   f"[{elapsed:4.1f}/{total:.0f}s]",
                   [base[0], base[1], base[2] - 0.205], (0.75, 0.78, 0.82), 0.95)

        if self.push_arrow is not None and sim_t > self.push_until:
            self._line("push", [0, 0, -5], [0, 0, -5], (0, 0, 0), 1.0)
            self.push_arrow = None

    def show_push(self, com, force, sim_t, hold=0.6):
        if not self.gui:
            return
        f = np.asarray(force, float)[:2]
        n = np.linalg.norm(f)
        if n < 1e-9:
            return
        d = f / n
        a = [com[0] - d[0] * 0.22, com[1] - d[1] * 0.22, 0.24]
        b = [com[0] - d[0] * 0.03, com[1] - d[1] * 0.03, 0.24]
        self._line("push", a, b, (1.0, 0.15, 0.9), 6.0)
        self.push_arrow = True
        self.push_until = sim_t + hold


class Demo:
    def __init__(self, gui=True, speed=1.0):
        self.sim = SimManager(gui=gui)
        self.rid = self.sim.init()
        self.robot = RobotInterface(self.rid)
        self.gui = gui
        self.dt = self.sim.timestep

        self.walk = WalkGenerator()
        self.walk.set_walk_parameter(**GAIT)
        self.walk.generate()
        self.walk.inverse_kinematics_all()

        self.overlay = Overlay(gui)
        self.pacer = Pacer(self.dt, speed)
        self.sim_t = 0.0
        self.ctrl = None
        self.pusher = None
        self.reset()

    def reset(self):
        """Put the robot back on its feet and rebuild the controller."""
        p.resetBasePositionAndOrientation(self.rid, [0, 0, SPAWN_Z],
                                          p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBaseVelocity(self.rid, [0, 0, 0], [0, 0, 0])
        self.ctrl = MainController(self.robot, self.walk, self.dt,
                                   sit=GAIT["sit"], enable_walk=False)
        self.robot.reset_motor_angles(self.ctrl.stand_pose)
        self.pusher = PushApplicator(self.robot, self.dt, duration=0.02)
        for _ in range(int(1.2 / self.dt)):
            self.ctrl.step()
            p.stepSimulation()

    def keys(self):
        """Returns 'next' / 'quit' / None."""
        if not self.gui:
            return None
        action = None
        for k, v in p.getKeyboardEvents().items():
            if not (v & p.KEY_WAS_TRIGGERED):
                continue
            shift = bool(p.getKeyboardEvents().get(p.B3G_SHIFT, 0)
                         & p.KEY_IS_DOWN)
            hard = 40 if shift else 18
            if k == p.B3G_UP_ARROW:
                self.fire("forward", hard)
            elif k == p.B3G_DOWN_ARROW:
                self.fire("backward", hard)
            elif k == p.B3G_LEFT_ARROW:
                self.fire("left", hard)
            elif k == p.B3G_RIGHT_ARROW:
                self.fire("right", hard)
            elif k == ord(" "):
                if self.ctrl.enable_walk:
                    self.ctrl.stop_walking()
                else:
                    self.ctrl.start_walking()
            elif k == ord("r"):
                self.reset()
            elif k == ord("n"):
                action = "next"
            elif k == ord("q"):
                action = "quit"
        return action

    def fire(self, direction, magnitude):
        ev = self.pusher.push(direction, magnitude)
        print(f"    t={self.sim_t:6.2f}s  PUSH {ev['label']}")
        self.overlay.show_push(self.robot.get_com_position(),
                               ev["force"], self.sim_t)

    def run_phase(self, phase):
        print(f"\n=== {phase['title']} ===\n    {phase['blurb']}")
        if phase["walk"]:
            self.ctrl.start_walking()
        else:
            self.ctrl.stop_walking()

        queue = sorted(phase["pushes"], key=lambda e: e[0])
        n = int(phase["secs"] / self.dt)
        t0 = self.sim_t
        draw_every = max(1, int(0.03 / self.dt))

        for i in range(n):
            if self.gui and not self.sim.is_connected():
                return "quit"
            act = self.keys()
            if act:
                return act

            elapsed = self.sim_t - t0
            while queue and queue[0][0] <= elapsed:
                _, d, m = queue.pop(0)
                self.fire(d, m)

            ev = self.pusher.update()
            if ev:
                self.overlay.show_push(self.robot.get_com_position(),
                                       ev["force"], self.sim_t)

            sense = self.ctrl.step()
            p.stepSimulation()
            self.sim_t += self.dt

            if i % draw_every == 0:
                self.overlay.draw(self.robot, sense, phase, elapsed,
                                  phase["secs"], self.ctrl.enable_walk,
                                  self.sim_t)
                self.sim.follow_camera(sense["com"])

            if sense["com"][2] < 0.13:
                print("    robot fell -- resetting")
                self.reset()
                if phase["walk"]:
                    self.ctrl.start_walking()
            self.pacer.tick()
        return None

    def free_play(self):
        phase = dict(title="FREE PLAY", walk=False, secs=0, pushes=[],
                     blurb="arrows = push, SHIFT+arrow = hard push, "
                           "SPACE = walk, R = reset, Q = quit")
        print(f"\n=== {phase['title']} ===\n    {phase['blurb']}")
        draw_every = max(1, int(0.03 / self.dt))
        i = 0
        while self.sim.is_connected():
            if self.keys() == "quit":
                return
            self.pusher.update()
            sense = self.ctrl.step()
            p.stepSimulation()
            self.sim_t += self.dt
            if i % draw_every == 0:
                self.overlay.draw(self.robot, sense, phase, 0.0, 0.0,
                                  self.ctrl.enable_walk, self.sim_t)
                self.sim.follow_camera(sense["com"])
            if sense["com"][2] < 0.13:
                print("    robot fell -- resetting")
                self.reset()
            i += 1
            self.pacer.tick()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", help="run only this phase "
                                    "(intro/ankle/hip/step/walk/walkpush)")
    ap.add_argument("--free", action="store_true",
                    help="skip the tour and go straight to manual control")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed (1.0 = wall clock)")
    ap.add_argument("--no-free", action="store_true",
                    help="exit after the tour instead of free play")
    ap.add_argument("--headless", action="store_true",
                    help="run without a window (smoke test)")
    ap.add_argument("--list", action="store_true", help="list phases and exit")
    args = ap.parse_args()

    if args.list:
        for ph in PHASES:
            print(f"  {ph['name']:<10} {ph['title']}")
        return 0

    demo = Demo(gui=not args.headless, speed=args.speed)
    print("\n" + "=" * 70)
    print("  Bipedal walking + push recovery -- visual showcase")
    print("  arrows = push   SHIFT+arrow = hard push   SPACE = walk")
    print("  R = reset       N = next phase            Q = quit")
    print("=" * 70)

    try:
        if not args.free:
            phases = PHASES
            if args.phase:
                phases = [ph for ph in PHASES if ph["name"] == args.phase]
                if not phases:
                    print(f"unknown phase '{args.phase}'; use --list")
                    return 1
            for ph in phases:
                if demo.run_phase(ph) == "quit":
                    break
                demo.reset()
        if not args.no_free and demo.sim.is_connected():
            demo.free_play()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        demo.sim.disconnect()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
