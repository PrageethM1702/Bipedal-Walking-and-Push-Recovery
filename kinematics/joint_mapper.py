"""
kinematics/joint_mapper.py

The inverse-kinematics module already emits the 12 leg angles in the exact
order the motors expect (right leg, then left leg), so "mapping" is mostly
about giving the balance strategies a readable way to add offsets to specific
joints of the 12-vector.

Layout of the 12-vector::

    index : 0        1         2          3     4            5
            R_hip_yaw R_hip_roll R_hip_pitch R_knee R_ankle_pitch R_ankle_roll
    index : 6        7         8          9    10           11
            L_hip_yaw L_hip_roll L_hip_pitch L_knee L_ankle_pitch L_ankle_roll
"""
import numpy as np

R_HIP_YAW, R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_ANKLE_PITCH, R_ANKLE_ROLL = range(0, 6)
L_HIP_YAW, L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE_PITCH, L_ANKLE_ROLL = range(6, 12)

ANKLE_PITCH = (R_ANKLE_PITCH, L_ANKLE_PITCH)
ANKLE_ROLL  = (R_ANKLE_ROLL,  L_ANKLE_ROLL)
HIP_PITCH   = (R_HIP_PITCH,   L_HIP_PITCH)
HIP_ROLL    = (R_HIP_ROLL,    L_HIP_ROLL)


def add_ankle_pitch(angles, d_right, d_left=None):
    d_left = d_right if d_left is None else d_left
    angles[R_ANKLE_PITCH] += d_right
    angles[L_ANKLE_PITCH] += d_left
    return angles


def add_ankle_roll(angles, d_right, d_left=None):
    d_left = d_right if d_left is None else d_left
    angles[R_ANKLE_ROLL] += d_right
    angles[L_ANKLE_ROLL] += d_left
    return angles


def add_hip_pitch(angles, d_right, d_left=None):
    d_left = d_right if d_left is None else d_left
    angles[R_HIP_PITCH] += d_right
    angles[L_HIP_PITCH] += d_left
    return angles
