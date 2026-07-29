"""
balance/ankle_strategy.py

Ankle strategy: the first line of defence against a disturbance.  Small
deviations of the trunk are corrected by ankle torque, which we realise on a
position-controlled robot as small offsets on the ankle pitch/roll joints.

The command is a PD law on the trunk tilt (and its rate), which keeps the
centre of pressure moving to oppose the fall while the capture point is still
inside the support polygon.  Because the feet are short, the ankle strategy has
a limited range; past it the controller escalates to the hip / stepping
strategies.
"""
import numpy as np

from kinematics import joint_mapper as jm


class AnkleStrategy:
    def __init__(self, kp_pitch=1.0, kd_pitch=0.03,
                 kp_roll=1.0, kd_roll=0.03, max_offset=0.35):
        self.kp_pitch = kp_pitch
        self.kd_pitch = kd_pitch
        self.kp_roll = kp_roll
        self.kd_roll = kd_roll
        self.max_offset = max_offset

    def compute(self, roll, pitch, roll_rate, pitch_rate):
        """Return (d_ankle_pitch, d_ankle_roll) offsets in radians.

        A positive trunk pitch (leaning forward) commands a positive ankle
        pitch offset, which drives the trunk back upright."""
        d_pitch = self.kp_pitch * pitch + self.kd_pitch * pitch_rate
        d_roll  = self.kp_roll * roll + self.kd_roll * roll_rate
        d_pitch = float(np.clip(d_pitch, -self.max_offset, self.max_offset))
        d_roll  = float(np.clip(d_roll,  -self.max_offset, self.max_offset))
        return d_pitch, d_roll

    def apply(self, angles, roll, pitch, roll_rate, pitch_rate):
        """Add the ankle offsets to a 12-vector of joint angles in place."""
        d_pitch, d_roll = self.compute(roll, pitch, roll_rate, pitch_rate)
        jm.add_ankle_pitch(angles, d_pitch)
        jm.add_ankle_roll(angles, d_roll)
        return angles
