"""
test_robot.py -- self-checks for the walking / push-recovery stack.

Runs headless and exercises the pieces end to end:

    python test_robot.py

Each check prints PASS/FAIL and the script exits non-zero if anything failed.
"""
import sys
import os

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.sim_manager import SimManager
from simulation.robot_interface import RobotInterface
from simulation.push_applicator import PushApplicator
from walking.walk_generator import WalkGenerator
from kinematics.inverse_kinematics import solve_leg_ik, solve_both_legs
from balance.capture_point import capture_point, omega
from balance.support_polygon import SupportPolygon
from balance.recovery_step import RecoveryStepPlanner
from control.main_controller import MainController
from control.state_machine import BalanceStateMachine, State

GAIT = dict(bodyMovePoint=8, legMovePoint=8, height=50, stride=90, sit=50,
            swayBody=30, swayFoot=0, bodyPositionForwardPlus=5, swayShift=3,
            liftPush=0.5, landPull=0.7, timeStep=0.06)

results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" +
          (f"  ({detail})" if detail else ""))


def test_kinematics():
    print("\nKinematics")
    a = solve_leg_ik((0, 0, 50), "right")
    check("IK returns 6 joint angles", a.shape == (6,))
    check("IK is finite", np.all(np.isfinite(a)))
    both = solve_both_legs((0, 0, 50), (0, 0, 50))
    check("both legs -> 12 angles", both.shape == (12,))
    fwd = solve_leg_ik((60, 0, 50), "right")
    check("stepping forward changes hip pitch", abs(fwd[2] - a[2]) > 1e-3,
          f"{a[2]:.3f} -> {fwd[2]:.3f}")
    far = solve_leg_ik((10_000, 0, 50), "right")
    check("unreachable target is clamped", np.all(np.isfinite(far)))


def test_capture_point():
    print("\nCapture point")
    w = omega(0.25)
    check("omega = sqrt(g/z)", abs(w - np.sqrt(9.81 / 0.25)) < 1e-9,
          f"{w:.3f} rad/s")
    cp = capture_point((0.0, 0.0), (0.0, 0.0), 0.25)
    check("zero velocity -> CP at the CoM", np.allclose(cp, [0, 0]))
    cp = capture_point((0.0, 0.0), (0.5, 0.0), 0.25)
    check("forward velocity moves CP forward", cp[0] > 0, f"cp_x={cp[0]:.3f}")


def test_support_polygon():
    print("\nSupport polygon")
    sp = SupportPolygon((0.0, 0.04), (0.0, -0.04))
    check("polygon has vertices", len(sp.vertices) >= 4)
    check("centre is inside", sp.contains((0.0, 0.0)))
    check("far point is outside", not sp.contains((1.0, 0.0)))
    check("margin positive inside", sp.margin((0.0, 0.0)) > 0)
    check("margin negative outside", sp.margin((1.0, 0.0)) < 0)
    single = SupportPolygon((0.0, 0.04), (0.0, -0.04), right_contact=False)
    check("single support is smaller",
          single.margin((0.0, 0.0)) < sp.margin((0.0, 0.0)))


def test_step_planner():
    print("\nRecovery step planner")
    pl = RecoveryStepPlanner(sit=50)
    wps = pl.plan_full((0.06, 0.0), 0.4)
    check("plan produced", len(wps) > 0, f"{len(wps)} way-points")
    xs = [float(r[0]) for r, l, d in wps] + [float(l[0]) for r, l, d in wps]
    check("forward push -> forward step", max(xs) > 0, f"max x={max(xs):.0f} mm")
    back = pl.plan_full((-0.06, 0.0), 0.4)
    bxs = [float(r[0]) for r, l, d in back] + [float(l[0]) for r, l, d in back]
    check("backward push -> backward step", min(bxs) < 0)
    check("backward step is clamped shorter",
          abs(min(bxs)) <= pl.max_back_step + 1e-6,
          f"{min(bxs):.0f} mm")
    left = pl.plan_full((0.0, 0.06), 0.4)
    check("left push leads with the left foot",
          pl.lead_foot((0.0, 0.06)) == "L")


def test_state_machine():
    print("\nState machine")
    fsm = BalanceStateMachine()
    check("settled -> STAND",
          fsm.update(0.05, 0.01, 0.01, com_z=0.23) == State.STAND)
    check("CP outside + fast -> STEP_RECOVER",
          fsm.update(-0.05, 0.6, 0.05, com_z=0.23) == State.STEP_RECOVER)
    fsm2 = BalanceStateMachine()
    check("collapsed -> FALLEN",
          fsm2.update(0.05, 0.0, 0.0, com_z=0.05) == State.FALLEN)
    fsm3 = BalanceStateMachine()
    check("walking is not interrupted by normal gait motion",
          fsm3.update(-0.05, 0.5, 0.1, com_z=0.23, walking=True) == State.WALK)


def _build(gui=False):
    sim = SimManager(gui=gui)
    rid = sim.init()
    robot = RobotInterface(rid)
    walk = WalkGenerator()
    walk.set_walk_parameter(**GAIT)
    walk.generate()
    walk.inverse_kinematics_all()
    ctrl = MainController(robot, walk, sim.timestep, sit=GAIT["sit"])
    robot.reset_motor_angles(ctrl.stand_pose)
    return sim, robot, walk, ctrl


def test_standing():
    print("\nStanding (integration)")
    sim, robot, walk, ctrl = _build()
    for _ in range(int(3.0 / sim.timestep)):
        ctrl.step()
        p.stepSimulation()
    com = robot.get_com_position()
    tilt = max(abs(robot.get_base_orientation()[0]),
               abs(robot.get_base_orientation()[1]))
    check("stays upright for 3 s", com[2] > 0.20, f"z={com[2]:.3f}")
    check("stays level", tilt < 0.10, f"tilt={tilt:.3f}")
    lz = robot.get_left_foot_pos()[2]
    rz = robot.get_right_foot_pos()[2]
    check("both feet planted", lz < 0.04 and rz < 0.04,
          f"lf_z={lz:.3f} rf_z={rz:.3f}")
    sim.disconnect()


def test_walking():
    print("\nWalking (integration)")
    sim, robot, walk, ctrl = _build()
    for _ in range(int(1.0 / sim.timestep)):
        ctrl.step()
        p.stepSimulation()
    start = robot.get_com_position()
    ctrl.start_walking()
    for _ in range(int(7.0 / sim.timestep)):
        ctrl.step()
        p.stepSimulation()
    end = robot.get_com_position()
    dist = end[0] - start[0]
    check("did not fall", end[2] > 0.20, f"z={end[2]:.3f}")
    check("walked forward", dist > 0.30, f"{dist:+.3f} m")
    check("tracked a straight line", abs(end[1] - start[1]) < 0.20,
          f"lateral drift {end[1]-start[1]:+.3f} m")
    sim.disconnect()


def test_push_recovery():
    print("\nPush recovery (integration)")
    for direction, mag in (("forward", 25), ("backward", 20),
                           ("left", 25), ("right", 25)):
        sim, robot, walk, ctrl = _build()
        for _ in range(int(1.0 / sim.timestep)):
            ctrl.step()
            p.stepSimulation()
        pusher = PushApplicator(robot, sim.timestep, duration=0.02)
        pusher.schedule(1.0, direction, mag)
        for _ in range(int(5.0 / sim.timestep)):
            pusher.update()
            s = ctrl.step()
            p.stepSimulation()
            if s["com"][2] < 0.13:
                break
        com = robot.get_com_position()
        tilt = max(abs(robot.get_base_orientation()[0]),
                   abs(robot.get_base_orientation()[1]))
        check(f"recovers a {mag} N push {direction}",
              com[2] > 0.20 and tilt < 0.4,
              f"z={com[2]:.3f} tilt={tilt:.2f}")
        sim.disconnect()


def main():
    print("=" * 62)
    print("Bipedal walking + push recovery -- self-checks")
    print("=" * 62)
    test_kinematics()
    test_capture_point()
    test_support_polygon()
    test_step_planner()
    test_state_machine()
    test_standing()
    test_walking()
    test_push_recovery()

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print("\n" + "=" * 62)
    print(f"{passed}/{total} checks passed")
    failed = [n for n, ok in results if not ok]
    if failed:
        print("failed:")
        for n in failed:
            print(f"   - {n}")
    print("=" * 62)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
