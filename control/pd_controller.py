"""
control/pd_controller.py

Low-level standing-posture controller: holds a target standing pose and adds the
continuous ankle stabilisation used to keep the robot upright when it is not
walking.  It is a thin convenience layer over the ankle strategy for demos /
tests that only need standing balance (the full behaviour lives in
``control.main_controller.MainController``).
"""
import numpy as np

from kinematics.inverse_kinematics import solve_both_legs
from balance.ankle_strategy import AnkleStrategy


class StandingController:
    def __init__(self, robot, dt, sit=50.0,
                 kp_pitch=1.0, kd_pitch=0.03, kp_roll=1.0, kd_roll=0.03):
        self.robot = robot
        self.dt = dt
        self.stand_pose = solve_both_legs((0, 0, sit), (0, 0, sit))
        self.ankle = AnkleStrategy(kp_pitch, kd_pitch, kp_roll, kd_roll)
        self._prev_rpy = robot.get_base_orientation()

    def reset_to_stand(self):
        self.robot.reset_motor_angles(self.stand_pose)

    def stabilize(self):
        """Compute and apply one tick of stabilised standing pose."""
        rpy = self.robot.get_base_orientation()
        rate = (rpy - self._prev_rpy) / self.dt
        self._prev_rpy = rpy
        angles = self.stand_pose.copy()
        self.ankle.apply(angles, rpy[0], rpy[1], rate[0], rate[1])
        self.robot.apply_motor_angles(angles)
        return angles
