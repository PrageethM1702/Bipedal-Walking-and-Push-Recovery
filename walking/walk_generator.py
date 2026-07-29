"""
walking/walk_generator.py

Feed-forward gait generator for the 12-DoF biped.

It builds smooth foot trajectories (sinusoidal swing + lateral body sway) for a
full walk cycle and converts them, via the leg inverse kinematics, into
sequences of 12 joint angles.  A cycle is split into:

    walkPoint0 : double support, foot that just landed
    walkPoint1 : single support, supporting foot
    walkPoint2 : double support, foot about to lift
    walkPoint3 : single support, swinging foot

plus dedicated *start* and *end* motions so the robot can begin and finish
walking from a standstill.

This is a headless (matplotlib-free) adaptation of Sunbin Kim's (Einsbon)
open-source walk generator; the trajectory maths is preserved because it is
known to produce a stable walk.  See README for attribution.

All lengths are in millimetres, matching the inverse-kinematics module.
"""
import math
import numpy as np

from kinematics.inverse_kinematics import solve_leg_ik


class WalkGenerator:
    def __init__(self):
        self._pelvic_interval  = 70.5
        self._legUp_length     = 110.0
        self._legDown_length   = 110.0
        self._footJoint_to_bottom = 45.0

    def set_walk_parameter(self, bodyMovePoint=8, legMovePoint=8, height=50,
                           stride=90, sit=50, swayBody=30, swayFoot=0,
                           bodyPositionForwardPlus=5, swayShift=3,
                           liftPush=0.5, landPull=0.7, timeStep=0.06,
                           damping=0.0, incline=0.0):
        self._bodyMovePoint = bodyMovePoint
        self._legMovePoint  = legMovePoint
        self._h             = height
        self._l             = stride
        self._sit           = sit
        self._swayBody      = swayBody
        self._swayFoot      = swayFoot
        self._swayShift     = swayShift
        self._liftPush      = liftPush
        self._landPull      = landPull
        self._timeStep      = timeStep
        self._bodyPositionXPlus = bodyPositionForwardPlus
        self._damping       = damping
        self._incline       = incline
        self._stepPoint     = bodyMovePoint + legMovePoint

    def generate(self):
        walkPoint = self._bodyMovePoint * 2 + self._legMovePoint * 2
        trajectoryLength = self._l * (2 * self._bodyMovePoint + self._legMovePoint) \
            / (self._bodyMovePoint + self._legMovePoint)

        walkPoint0 = np.zeros((3, self._bodyMovePoint))
        walkPoint1 = np.zeros((3, self._legMovePoint))
        walkPoint2 = np.zeros((3, self._bodyMovePoint))
        walkPoint3 = np.zeros((3, self._legMovePoint))

        self.walkPointStartRightstepRightLeg = np.zeros((3, self._bodyMovePoint + self._legMovePoint))
        self.walkPointStartLeftstepRightLeg  = np.zeros((3, self._bodyMovePoint + self._legMovePoint))
        self.walkPointEndRightstepRightLeg   = np.zeros((3, self._bodyMovePoint + self._legMovePoint))
        self.walkPointEndLeftstepRightLeg    = np.zeros((3, self._bodyMovePoint + self._legMovePoint))

        for i in range(self._bodyMovePoint):
            t = (i + 1) / (walkPoint - self._legMovePoint)
            walkPoint0[0][i] = -trajectoryLength * (t - 0.5)
            walkPoint0[2][i] = self._sit
            walkPoint0[1][i] = self._swayBody * math.sin(2 * math.pi * ((i + 1 - self._swayShift) / walkPoint))

        for i in range(self._legMovePoint):
            t = (i + 1 + self._bodyMovePoint) / (walkPoint - self._legMovePoint)
            walkPoint1[0][i] = -trajectoryLength * (t - 0.5)
            walkPoint1[2][i] = self._sit
            walkPoint1[1][i] = self._swayBody * math.sin(2 * math.pi * ((i + 1 + self._bodyMovePoint - self._swayShift) / walkPoint))

        for i in range(self._bodyMovePoint):
            t = (i + 1 + self._bodyMovePoint + self._legMovePoint) / (walkPoint - self._legMovePoint)
            walkPoint2[0][i] = -trajectoryLength * (t - 0.5)
            walkPoint2[2][i] = self._sit
            walkPoint2[1][i] = self._swayBody * math.sin(2 * math.pi * ((i + 1 + self._bodyMovePoint + self._legMovePoint - self._swayShift) / walkPoint))

        for i in range(self._legMovePoint):
            t = (i + 1) / self._legMovePoint
            sin_tpi = math.sin(t * math.pi)
            walkPoint3[0][i] = (2 * t - 1 + (1 - t) * self._liftPush * -sin_tpi + t * self._landPull * sin_tpi) * trajectoryLength / 2
            walkPoint3[2][i] = math.sin(t * math.pi) * self._h + self._sit
            walkPoint3[1][i] = math.sin(t * math.pi) * self._swayFoot + self._swayBody * math.sin(2 * math.pi * ((i + 1 + walkPoint - self._legMovePoint - self._swayShift) / walkPoint))

        for i in range(self._bodyMovePoint - self._swayShift):
            self.walkPointStartRightstepRightLeg[2][i] = self._sit
            self.walkPointStartLeftstepRightLeg[2][i]  = self._sit

        for i in range(self._legMovePoint):
            t = (i + 1) / self._legMovePoint
            t2 = (i + 1) / (self._legMovePoint + self._swayShift)
            sin_tpi = math.sin(t * math.pi)
            k = i + self._bodyMovePoint - self._swayShift
            self.walkPointStartRightstepRightLeg[2][k] = math.sin(t * math.pi) * self._h + self._sit
            self.walkPointStartRightstepRightLeg[0][k] = (2 * t + (1 - t) * self._liftPush * -sin_tpi + t * self._landPull * sin_tpi) * trajectoryLength / 4
            self.walkPointStartLeftstepRightLeg[0][k] = (math.cos(t2 * math.pi / 2) - 1) * trajectoryLength * ((self._swayShift + self._bodyMovePoint + self._legMovePoint) / (self._bodyMovePoint * 2 + self._legMovePoint) - 0.5)
            self.walkPointStartLeftstepRightLeg[2][k] = self._sit

        for i in range(self._swayShift):
            t2 = (i + 1 + self._legMovePoint) / (self._legMovePoint + self._swayShift)
            k = i + self._legMovePoint + self._bodyMovePoint - self._swayShift
            self.walkPointStartRightstepRightLeg[0][k] = -trajectoryLength * ((i + 1) / (walkPoint - self._legMovePoint) - 0.5)
            self.walkPointStartRightstepRightLeg[2][k] = self._sit
            self.walkPointStartLeftstepRightLeg[0][k] = (math.cos(t2 * math.pi / 2) - 1) * trajectoryLength * ((self._swayShift + self._bodyMovePoint + self._legMovePoint) / (self._bodyMovePoint * 2 + self._legMovePoint) - 0.5)
            self.walkPointStartLeftstepRightLeg[2][k] = self._sit

        for i in range(self._bodyMovePoint + self._legMovePoint):
            t = (i + 1) / (self._bodyMovePoint + self._legMovePoint)
            if t < 1 / 4:
                self.walkPointStartRightstepRightLeg[1][i] = -self._swayBody * (math.sin(t * math.pi) - (1 - math.sin(math.pi * 2 * t)) * (math.sin(4 * t * math.pi) / 4))
                self.walkPointStartLeftstepRightLeg[1][i]  = self._swayBody * (math.sin(t * math.pi) - (1 - math.sin(math.pi * 2 * t)) * (math.sin(4 * t * math.pi) / 4))
            else:
                self.walkPointStartRightstepRightLeg[1][i] = -self._swayBody * math.sin(t * math.pi)
                self.walkPointStartLeftstepRightLeg[1][i]  = self._swayBody * math.sin(t * math.pi)

        for i in range(self._bodyMovePoint - self._swayShift):
            self.walkPointEndLeftstepRightLeg[0][i] = -trajectoryLength * ((i + 1 + self._swayShift) / (walkPoint - self._legMovePoint) - 0.5)
            self.walkPointEndLeftstepRightLeg[2][i] = self._sit
            self.walkPointEndRightstepRightLeg[0][i] = -trajectoryLength * ((i + 1 + self._swayShift + self._bodyMovePoint + self._legMovePoint) / (walkPoint - self._legMovePoint) - 0.5)
            self.walkPointEndRightstepRightLeg[2][i] = self._sit
        for i in range(self._legMovePoint):
            t = (i + 1) / self._legMovePoint
            sin_tpi = math.sin(t * math.pi)
            k = i + self._bodyMovePoint - self._swayShift
            self.walkPointEndLeftstepRightLeg[0][k] = (math.sin(t * math.pi / 2) - 1) * trajectoryLength * ((self._bodyMovePoint) / (self._bodyMovePoint * 2 + self._legMovePoint) - 0.5)
            self.walkPointEndLeftstepRightLeg[2][k] = self._sit
            self.walkPointEndRightstepRightLeg[0][k] = (2 * t - 2 + (1 - t) * self._liftPush * -sin_tpi + t * self._landPull * sin_tpi) * trajectoryLength / 4
            self.walkPointEndRightstepRightLeg[2][k] = math.sin(t * math.pi) * self._h + self._sit
        for i in range(self._swayShift):
            k = i + self._bodyMovePoint + self._legMovePoint - self._swayShift
            self.walkPointEndLeftstepRightLeg[2][k] = self._sit
            self.walkPointEndRightstepRightLeg[2][k] = self._sit

        for i in range(self._bodyMovePoint + self._legMovePoint):
            t = 1 - (i + 1) / (self._bodyMovePoint + self._legMovePoint)
            if t < 1 / 4:
                self.walkPointEndLeftstepRightLeg[1][i]  = self._swayBody * (math.sin(t * math.pi) - (1 - math.sin(math.pi * 2 * t)) * (math.sin(4 * t * math.pi) / 4))
                self.walkPointEndRightstepRightLeg[1][i] = -self._swayBody * (math.sin(t * math.pi) - (1 - math.sin(math.pi * 2 * t)) * (math.sin(4 * t * math.pi) / 4))
            else:
                self.walkPointEndLeftstepRightLeg[1][i]  = self._swayBody * math.sin(t * math.pi)
                self.walkPointEndRightstepRightLeg[1][i] = -self._swayBody * math.sin(t * math.pi)

        if self._incline != 0:
            for wp in (walkPoint0, walkPoint1, walkPoint2, walkPoint3):
                wp[2] = wp[2] + wp[0] * self._incline
            for wp in (self.walkPointStartRightstepRightLeg, self.walkPointStartLeftstepRightLeg,
                       self.walkPointEndLeftstepRightLeg, self.walkPointEndRightstepRightLeg):
                wp[2] = wp[2] + wp[0] * self._incline

        if self._bodyPositionXPlus != 0:
            for wp in (walkPoint0, walkPoint1, walkPoint2, walkPoint3,
                       self.walkPointStartRightstepRightLeg, self.walkPointStartLeftstepRightLeg,
                       self.walkPointEndLeftstepRightLeg, self.walkPointEndRightstepRightLeg):
                wp[0] = wp[0] - self._bodyPositionXPlus

        if self._damping != 0:
            dampHeight = (walkPoint3[2][-1] - walkPoint0[2][0]) / 2
            walkPoint0[2][0] += dampHeight * self._damping
            walkPoint2[2][0] -= dampHeight * self._damping

        self._walkPoint0 = walkPoint0
        self._walkPoint1 = walkPoint1
        self._walkPoint2 = walkPoint2
        self._walkPoint3 = walkPoint3

        s = self._swayShift
        self.walkPointLeftStepRightLeg  = np.column_stack([walkPoint0[:, s:], walkPoint1, walkPoint2[:, :s]])
        self.walkPointRightStepRightLeg = np.column_stack([walkPoint2[:, s:], walkPoint3, walkPoint0[:, :s]])

        mirror = np.array([[1], [-1], [1]])
        self.walkPointLeftStepLeftLeg  = self.walkPointRightStepRightLeg * mirror
        self.walkPointRightStepLeftLeg = self.walkPointLeftStepRightLeg * mirror
        self.walkPointStartRightstepLeftLeg = self.walkPointStartLeftstepRightLeg * mirror
        self.walkPointStartLeftstepLeftLeg  = self.walkPointStartRightstepRightLeg * mirror
        self.walkPointEndLeftstepLeftLeg  = self.walkPointEndRightstepRightLeg * mirror
        self.walkPointEndRightstepLeftLeg = self.walkPointEndLeftstepRightLeg * mirror

    def _ik_list(self, point, side):
        n = point[0].size
        out = np.zeros((n, 6))
        for i in range(n):
            out[i] = solve_leg_ik((point[0][i], point[1][i], point[2][i]), side)
        return out

    def inverse_kinematics_all(self):
        self.walkAnglesStartRight = np.column_stack(
            [self._ik_list(self.walkPointStartRightstepRightLeg, "right"),
             self._ik_list(self.walkPointStartRightstepLeftLeg,  "left")])
        self.walkAnglesStartLeft = np.column_stack(
            [self._ik_list(self.walkPointStartLeftstepRightLeg, "right"),
             self._ik_list(self.walkPointStartLeftstepLeftLeg,  "left")])
        self.walkAnglesEndLeft = np.column_stack(
            [self._ik_list(self.walkPointEndLeftstepRightLeg, "right"),
             self._ik_list(self.walkPointEndLeftstepLeftLeg,  "left")])
        self.walkAnglesEndRight = np.column_stack(
            [self._ik_list(self.walkPointEndRightstepRightLeg, "right"),
             self._ik_list(self.walkPointEndRightstepLeftLeg,  "left")])
        self.walkAnglesWalkingRight = np.column_stack(
            [self._ik_list(self.walkPointRightStepRightLeg, "right"),
             self._ik_list(self.walkPointRightStepLeftLeg,  "left")])
        self.walkAnglesWalkingLeft = np.column_stack(
            [self._ik_list(self.walkPointLeftStepRightLeg, "right"),
             self._ik_list(self.walkPointLeftStepLeftLeg,  "left")])

    def standing_angles(self):
        return solve_both_pose(self._sit)


def solve_both_pose(sit):
    """Neutral standing pose at the given sit height (both feet under hips)."""
    from kinematics.inverse_kinematics import solve_both_legs
    return solve_both_legs((0, 0, sit), (0, 0, sit))
