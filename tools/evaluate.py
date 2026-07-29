"""
tools/evaluate.py

Headless evaluation harness: pushes the robot from every direction over a range
of magnitudes and reports which ones it recovers from.  This is what the
controller gains were tuned against, and it is the quickest way to check that a
change has not regressed the balance behaviour.

Usage:
    python tools/evaluate.py                 # standing push battery
    python tools/evaluate.py --walking       # push it while it walks
    python tools/evaluate.py --mags 20 30 40 # custom magnitudes
"""
import argparse
import os
import sys

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.sim_manager import SimManager
from simulation.robot_interface import RobotInterface
from simulation.push_applicator import PushApplicator
from walking.walk_generator import WalkGenerator
from control.main_controller import MainController

GAIT = dict(bodyMovePoint=8, legMovePoint=8, height=50, stride=90, sit=50,
            swayBody=30, swayFoot=0, bodyPositionForwardPlus=5, swayShift=3,
            liftPush=0.5, landPull=0.7, timeStep=0.06)

DIRECTIONS = ["forward", "backward", "left", "right"]


def run_trial(direction, magnitude, walking=False, settle=1.0,
              push_at=2.0, horizon=6.0, push_duration=0.02):
    """Returns (recovered, peak_tilt, stepped)."""
    sim = SimManager(gui=False)
    rid = sim.init()
    robot = RobotInterface(rid)

    walk = WalkGenerator()
    walk.set_walk_parameter(**GAIT)
    walk.generate()
    walk.inverse_kinematics_all()

    ctrl = MainController(robot, walk, sim.timestep, sit=GAIT["sit"])
    robot.reset_motor_angles(ctrl.stand_pose)

    for _ in range(int(settle / sim.timestep)):
        ctrl.step()
        p.stepSimulation()
    if walking:
        ctrl.start_walking()
        for _ in range(int(2.0 / sim.timestep)):
            ctrl.step()
            p.stepSimulation()

    pusher = PushApplicator(robot, sim.timestep, duration=push_duration)
    pusher.schedule(push_at, direction, magnitude)

    peak_tilt = 0.0
    stepped = False
    fell = False
    for _ in range(int(horizon / sim.timestep)):
        pusher.update()
        s = ctrl.step()
        p.stepSimulation()
        peak_tilt = max(peak_tilt, s["tilt"])
        if s["state"].value == "STEP_RECOVER":
            stepped = True
        if s["com"][2] < 0.13:
            fell = True
            break

    final_tilt = max(abs(robot.get_base_orientation()[0]),
                     abs(robot.get_base_orientation()[1]))
    sim.disconnect()
    recovered = (not fell) and final_tilt < 0.4
    return recovered, peak_tilt, stepped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walking", action="store_true",
                    help="push the robot while it is walking")
    ap.add_argument("--mags", type=float, nargs="+",
                    default=[10, 20, 25, 30, 35, 40, 45],
                    help="push magnitudes in newtons")
    ap.add_argument("--directions", nargs="+", default=DIRECTIONS)
    args = ap.parse_args()

    mode = "WALKING" if args.walking else "STANDING"
    print(f"\nPush-recovery evaluation -- {mode}")
    print(f"{'dir':<10}" + "".join(f"{m:>7.0f}N" for m in args.mags))
    print("-" * (10 + 8 * len(args.mags)))

    limits = {}
    for d in args.directions:
        cells, best = [], 0.0
        for m in args.mags:
            ok, tilt, stepped = run_trial(d, m, walking=args.walking)
            mark = "ok" if ok else "--"
            if ok and stepped:
                mark = "OK*"
            if ok:
                best = m
            cells.append(f"{mark:>8}")
        limits[d] = best
        print(f"{d:<10}" + "".join(cells))

    print("\n* = a protective step was required (not just ankle/hip)")
    print("Largest recovered push per direction:")
    for d, m in limits.items():
        print(f"   {d:<10} {m:>5.0f} N")


if __name__ == "__main__":
    main()
