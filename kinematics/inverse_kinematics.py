"""
kinematics/inverse_kinematics.py

Analytic inverse kinematics for one 6-DoF leg of the 12-DoF biped.

Given a desired foot position expressed **relative to the hip** (millimetres,
robot frame: +x forward, +y left, +z up) it returns the six joint angles

    [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]

The thigh and shank are modelled as a 2-link chain; the hip-roll / ankle-roll
pair handle the frontal plane and keep the foot flat.

The formulation follows the working walk controller by Sunbin Kim (Einsbon),
adapted here into a reusable module.  See README for attribution.
"""
import numpy as np

LEG_UP_LENGTH   = 110.0
LEG_DOWN_LENGTH = 110.0
FOOT_TO_GROUND  = 45.0
PELVIC_INTERVAL = 70.5

MOTOR_DIR_RIGHT = np.array([+1, +1, +1, +1, +1, +1])
MOTOR_DIR_LEFT  = np.array([+1, +1, +1, +1, +1, +1])


def solve_leg_ik(foot_point, side="right"):
    """
    foot_point : (x, y, z) desired foot position relative to the hip, in mm.
                 z is the *upward* displacement toward the hip (0 = fully
                 extended reach ``LEG_UP_LENGTH + LEG_DOWN_LENGTH``; larger z
                 means a more folded / "sitting" leg).
    side       : "right" or "left".
    Returns np.array of 6 joint angles (rad):
        [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]
    """
    l3, l4 = LEG_UP_LENGTH, LEG_DOWN_LENGTH
    fx = foot_point[0]
    fy = foot_point[1]
    fz = l3 + l4 - foot_point[2]

    a = np.sqrt(fx * fx + fy * fy + fz * fz)
    a = min(a, l3 + l4 - 1e-3)

    d1 = np.arcsin(np.clip(fx / a, -1.0, 1.0))
    d2 = np.arccos(np.clip((l3 * l3 + a * a - l4 * l4) / (2 * l3 * a), -1, 1))
    d4 = np.arccos(np.clip((l4 * l4 + a * a - l3 * l3) / (2 * l4 * a), -1, 1))
    d5 = np.pi - d2 - d4

    t1 = np.arctan2(fy, fz)
    t2 = d1 + d2
    t3 = np.pi - d5
    t4 = -t2 + t3
    t5 = -t1

    direction = MOTOR_DIR_RIGHT if side == "right" else MOTOR_DIR_LEFT
    return np.array([0.0, t1, -t2, t3, -t4, t5]) * direction


def solve_both_legs(right_point, left_point):
    """Return the concatenated 12-vector [right(6), left(6)]."""
    return np.hstack([solve_leg_ik(right_point, "right"),
                      solve_leg_ik(left_point, "left")])
