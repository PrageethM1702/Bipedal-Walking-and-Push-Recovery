"""
run_simulation.py -- main entry point.

Runs the 12-DoF biped in PyBullet: it walks, gets pushed, and recovers.
Everything is logged to ``logs/<run>/`` and (unless --no-plots) turned into
figures, including the planned-vs-actual footstep map.

Examples
--------
    python run_simulation.py                       # walk + scripted pushes (GUI)
    python run_simulation.py --scenario stand      # standing push recovery
    python run_simulation.py --scenario walk       # undisturbed walking
    python run_simulation.py --headless            # no window (fast)
    python run_simulation.py --push 30 --dir left  # one custom push
    python run_simulation.py --random-pushes       # random disturbances

Keyboard (GUI): arrow keys push the robot, SPACE toggles walking.
"""
import argparse
import os
import sys

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.sim_manager import SimManager
from simulation.robot_interface import RobotInterface
from simulation.push_applicator import PushApplicator
from walking.walk_generator import WalkGenerator
from control.main_controller import MainController
from tools.data_logger import DataLogger

GAIT = dict(bodyMovePoint=8, legMovePoint=8, height=50, stride=90, sit=50,
            swayBody=30, swayFoot=0, bodyPositionForwardPlus=5, swayShift=3,
            liftPush=0.5, landPull=0.7, timeStep=0.06)

SETTLE_TIME = 1.0


def build(gui=True):
    sim = SimManager(gui=gui)
    rid = sim.init()
    robot = RobotInterface(rid)

    walk = WalkGenerator()
    walk.set_walk_parameter(**GAIT)
    walk.generate()
    walk.inverse_kinematics_all()

    ctrl = MainController(robot, walk, sim.timestep,
                          sit=GAIT["sit"], enable_walk=False)
    robot.reset_motor_angles(ctrl.stand_pose)
    return sim, robot, walk, ctrl


def keyboard_pushes(pusher, ctrl):
    """Arrow keys push; SPACE toggles walking (GUI only)."""
    for k, v in p.getKeyboardEvents().items():
        if not (v & p.KEY_WAS_TRIGGERED):
            continue
        if k == p.B3G_UP_ARROW:
            pusher.push("forward", 25)
        elif k == p.B3G_DOWN_ARROW:
            pusher.push("backward", 20)
        elif k == p.B3G_LEFT_ARROW:
            pusher.push("left", 25)
        elif k == p.B3G_RIGHT_ARROW:
            pusher.push("right", 25)
        elif k == ord(" "):
            if ctrl.enable_walk:
                ctrl.stop_walking()
            else:
                ctrl.start_walking()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="walk_push",
                    choices=["walk_push", "stand", "walk", "step_recover",
                             "custom"],
                    help="which demo to run (default: walk_push)")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to simulate (default: scenario-dependent)")
    ap.add_argument("--headless", action="store_true", help="no GUI window")
    ap.add_argument("--realtime", action="store_true",
                    help="throttle the GUI to wall-clock speed")
    ap.add_argument("--push", type=float, default=25.0,
                    help="push magnitude in N for --scenario custom")
    ap.add_argument("--dir", default="forward",
                    choices=["forward", "backward", "left", "right"],
                    help="push direction for --scenario custom")
    ap.add_argument("--push-time", type=float, default=2.5,
                    help="when the custom push fires (s)")
    ap.add_argument("--random-pushes", action="store_true",
                    help="apply random pushes throughout the run")
    ap.add_argument("--run-name", default=None, help="log directory name")
    ap.add_argument("--no-log", action="store_true", help="disable logging")
    ap.add_argument("--no-plots", action="store_true",
                    help="log but do not render figures")
    args = ap.parse_args()

    gui = not args.headless
    sim, robot, walk, ctrl = build(gui=gui)
    dt = sim.timestep
    pusher = PushApplicator(robot, dt, duration=0.02)

    scenario = args.scenario
    duration = args.duration
    if scenario == "stand":
        duration = duration or 14.0
        for i, (d, m) in enumerate([("forward", 25), ("backward", 20),
                                    ("left", 30), ("right", 30)]):
            pusher.schedule(SETTLE_TIME + 1.5 + i * 3.0, d, m)
    elif scenario == "step_recover":
        duration = duration or 20.0
        for i, (d, m) in enumerate([("forward", 45), ("left", 30),
                                    ("right", 30)]):
            pusher.schedule(SETTLE_TIME + 1.5 + i * 6.0, d, m)
    elif scenario == "walk":
        duration = duration or 10.0
    elif scenario == "custom":
        duration = duration or 8.0
        pusher.schedule(SETTLE_TIME + args.push_time, args.dir, args.push)
    else:
        duration = duration or 14.0
        for i, (d, m) in enumerate([("left", 15), ("right", 15),
                                    ("backward", 15), ("left", 12)]):
            pusher.schedule(SETTLE_TIME + 3.0 + i * 2.4, d, m)
    if args.random_pushes:
        pusher.enable_random(interval=(2.5, 4.5), magnitude=(12, 28))

    logger = None if args.no_log else DataLogger(
        robot, ctrl, dt, run_name=args.run_name)

    for _ in range(int(SETTLE_TIME / dt)):
        ctrl.step()
        sim.step_realtime() if (gui and args.realtime) else sim.step()

    start_com = robot.get_com_position()
    if scenario in ("walk", "walk_push"):
        ctrl.start_walking()

    print(f"[run] scenario={scenario} duration={duration}s "
          f"gui={gui} logging={'off' if args.no_log else logger.dir}")
    if gui:
        print("[run] arrows = push, SPACE = toggle walking")

    fell = False
    n = int(duration / dt)
    for i in range(n):
        if gui and not sim.is_connected():
            break
        if gui:
            keyboard_pushes(pusher, ctrl)

        event = pusher.update()
        if event:
            print(f"  t={i*dt:6.2f}s  PUSH {event['label']}")

        sense = ctrl.step()
        if logger:
            logger.log(sense, push_event=event)

        sim.step_realtime() if (gui and args.realtime) else sim.step()

        if gui and i % 40 == 0:
            sim.follow_camera(sense["com"])
        if sense["com"][2] < 0.13:
            print(f"  t={i*dt:6.2f}s  ROBOT FELL")
            fell = True
            break

    end_com = robot.get_com_position()
    dist = float(end_com[0] - start_com[0])
    print(f"[run] finished: fell={fell}  forward distance={dist:+.3f} m")

    if logger:
        extra = dict(scenario=scenario, walked_x=round(dist, 4), fell=fell)
        log_dir, summary = logger.save(extra_summary=extra)
        print(f"[run] logs -> {log_dir}")
        if not args.no_plots:
            from tools.plot_logs import make_plots
            make_plots(log_dir)

    if gui:
        print("[run] simulation done -- close the window to exit.")
        while sim.is_connected():
            ctrl.step()
            sim.step_realtime()
    sim.disconnect()


if __name__ == "__main__":
    main()
