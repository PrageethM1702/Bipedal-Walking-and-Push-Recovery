"""
balance/hip_strategy.py

Hip strategy: the second line of defence, used for medium disturbances that
exceed the ankle's range but do not yet require a step.

The robot rapidly flexes the hips to accelerate the trunk against the fall.
This generates angular momentum about the CoM which shifts the Centroidal
Moment Pivot and produces a horizontal ground-reaction force that decelerates
the CoM -- the same principle you use when you windmill your arms to keep your
balance.

On this leg-only robot the "trunk" is the pelvis, so the strategy pitches/rolls
the pelvis by offsetting both hip-pitch / hip-roll joints, then relaxes.
"""
import numpy as np

from kinematics import joint_mapper as jm


class HipStrategy:
    def __init__(self, k_pitch=0.15, k_roll=0.15, max_offset=0.5,
                 relax=0.9):
        self.k_pitch = k_pitch
        self.k_roll = k_roll
        self.max_offset = max_offset
        self.relax = relax
        self._d_pitch = 0.0
        self._d_roll = 0.0

    def compute(self, com_vel_xy):
        """Return (d_hip_pitch, d_hip_roll) offsets (rad) from CoM velocity.

        Falling forward (+vx) bends the hips to throw the pelvis back."""
        vx, vy = com_vel_xy
        target_pitch = float(np.clip(-self.k_pitch * vx,
                                     -self.max_offset, self.max_offset))
        target_roll  = float(np.clip(self.k_roll * vy,
                                     -self.max_offset, self.max_offset))
        self._d_pitch = self.relax * self._d_pitch + (1 - self.relax) * target_pitch
        self._d_roll  = self.relax * self._d_roll + (1 - self.relax) * target_roll
        return self._d_pitch, self._d_roll

    def apply(self, angles, com_vel_xy):
        d_pitch, d_roll = self.compute(com_vel_xy)
        jm.add_hip_pitch(angles, d_pitch)
        return angles

    def reset(self):
        self._d_pitch = 0.0
        self._d_roll = 0.0
