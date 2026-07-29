"""
balance/capture_point.py

The Capture Point (a.k.a. instantaneous Capture Point / Divergent Component of
Motion) is the point on the ground where the robot must place its foot to come
to a complete stop, under the Linear Inverted Pendulum Model:

        xi = x_com + x_com_dot / omega ,      omega = sqrt(g / z_com)

If the capture point lies inside the support polygon the robot can stop without
stepping (ankle / hip strategies suffice).  If it lies outside, a step toward
the capture point is required.

Reference: Pratt et al., "Capture Point: A Step toward Humanoid Push Recovery",
IEEE-RAS Humanoids 2006.
"""
import numpy as np

G = 9.81


def omega(z_com, g=G):
    """Natural frequency of the LIPM for a CoM height ``z_com`` (m)."""
    return np.sqrt(g / max(z_com, 1e-3))


def capture_point(com_xy, com_vel_xy, z_com, g=G):
    """
    com_xy      : (x, y) CoM position on the ground plane (m).
    com_vel_xy  : (x, y) CoM horizontal velocity (m/s).
    z_com       : CoM height (m).
    Returns the capture point (x, y) in world coordinates (m).
    """
    com_xy = np.asarray(com_xy, dtype=float)
    com_vel_xy = np.asarray(com_vel_xy, dtype=float)
    return com_xy + com_vel_xy / omega(z_com, g)


def capture_point_offset(com_vel_xy, z_com, g=G):
    """Just the velocity-dependent offset xi - x_com (m)."""
    return np.asarray(com_vel_xy, dtype=float) / omega(z_com, g)


class CapturePointEstimator:
    """Convenience wrapper that reads the robot state and reports the CP."""

    def __init__(self, robot, g=G):
        self.robot = robot
        self.g = g

    def compute(self):
        com = self.robot.get_com_position()
        vel = self.robot.get_com_velocity()
        z = max(com[2], 1e-3)
        cp = capture_point(com[:2], vel[:2], z, self.g)
        return cp, com, vel
