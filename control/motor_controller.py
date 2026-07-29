"""
control/motor_controller.py

Position-controlled motor driver with smooth interpolation between joint
targets, used for the discrete start/stop walk motions and any scripted pose
transitions.  Adapted from Sunbin Kim's (Einsbon) MotorController.
"""
import numpy as np
import pybullet as p

from simulation.robot_interface import (MOTOR_JOINTS, MOTOR_KP, MOTOR_KD,
                                         MOTOR_TORQUE, MOTOR_MAX_VEL)


class MotorController:
    def __init__(self, robot_id, timestep,
                 kp=MOTOR_KP, kd=MOTOR_KD,
                 torque=MOTOR_TORQUE, max_velocity=MOTOR_MAX_VEL):
        self.robot_id = robot_id
        self.timestep = timestep
        self.kp = kp
        self.kd = kd
        self.torque = torque
        self.max_velocity = max_velocity
        self._target = np.array([p.getJointState(robot_id, j)[0]
                                 for j in MOTOR_JOINTS], dtype=float)

    def _command(self, angles):
        for idx, angle in zip(MOTOR_JOINTS, angles):
            p.setJointMotorControl2(
                self.robot_id, idx, controlMode=p.POSITION_CONTROL,
                targetPosition=float(angle), positionGain=self.kp,
                velocityGain=self.kd, force=self.torque,
                maxVelocity=self.max_velocity)

    def hold(self, steps=1):
        for _ in range(steps):
            self._command(self._target)
            p.stepSimulation()

    def move_to(self, angles, action_time, on_tick=None):
        """Interpolate from the current target to ``angles`` over
        ``action_time`` seconds, stepping the simulation each tick.  ``on_tick``
        (if given) is called every tick for pushes / logging."""
        angles = np.array(angles, dtype=float)
        if action_time <= 0:
            self._target = angles
            self._command(angles)
            p.stepSimulation()
            if on_tick:
                on_tick()
            return
        start = self._target.copy()
        n = max(1, int(action_time / self.timestep))
        for s in range(n):
            a = (s + 1) / n
            self._command(start + a * (angles - start))
            p.stepSimulation()
            if on_tick:
                on_tick()
        self._target = angles

    def play_sequence(self, frames, action_time, on_tick=None):
        for f in frames:
            self.move_to(f, action_time, on_tick=on_tick)
