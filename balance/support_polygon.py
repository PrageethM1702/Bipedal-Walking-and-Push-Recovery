"""
balance/support_polygon.py

Builds the support polygon from the feet that are currently in contact with
the ground and answers the geometric queries the balance controller needs:

  * is a point (e.g. the capture point) inside the support polygon?
  * signed distance from a point to the polygon boundary (+ inside),
  * the polygon centroid.

Each foot is modelled as an axis-aligned rectangle around the foot link, using
the half-extents measured from the robot model.  With one or two rectangles the
support polygon is their convex hull.
"""
import numpy as np

from simulation.robot_interface import FOOT_HALF_LEN, FOOT_HALF_WID


def _foot_corners(foot_xy, half_len=FOOT_HALF_LEN, half_wid=FOOT_HALF_WID):
    x, y = foot_xy
    return [(x - half_len, y - half_wid), (x + half_len, y - half_wid),
            (x + half_len, y + half_wid), (x - half_len, y + half_wid)]


def _convex_hull(points):
    """Andrew's monotone chain convex hull. Returns CCW list of vertices."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


class SupportPolygon:
    def __init__(self, left_foot_xy=None, right_foot_xy=None,
                 left_contact=True, right_contact=True):
        corners = []
        if left_foot_xy is not None and left_contact:
            corners += _foot_corners(left_foot_xy)
        if right_foot_xy is not None and right_contact:
            corners += _foot_corners(right_foot_xy)
        self.vertices = _convex_hull(corners) if corners else []

    @classmethod
    def from_robot(cls, robot):
        lc, rc = robot.get_foot_contacts()
        lf = robot.get_left_foot_pos()[:2]
        rf = robot.get_right_foot_pos()[:2]
        if not (lc or rc):
            lc = rc = True
        return cls(lf, rf, lc, rc)

    def centroid(self):
        if not self.vertices:
            return np.zeros(2)
        return np.mean(np.array(self.vertices), axis=0)

    def contains(self, point):
        """True if ``point`` (x, y) is inside the (convex) polygon."""
        v = self.vertices
        if len(v) < 3:
            return False
        px, py = point
        sign = None
        n = len(v)
        for i in range(n):
            ax, ay = v[i]
            bx, by = v[(i + 1) % n]
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            if abs(cross) < 1e-12:
                continue
            s = cross > 0
            if sign is None:
                sign = s
            elif s != sign:
                return False
        return True

    def signed_distance(self, point):
        """Distance to the boundary; positive inside, negative outside (m)."""
        v = self.vertices
        if len(v) < 3:
            return -np.linalg.norm(np.asarray(point) - self.centroid())
        px, py = point
        min_d = float("inf")
        n = len(v)
        for i in range(n):
            a = np.array(v[i]); b = np.array(v[(i + 1) % n])
            ab = b - a; t = np.clip(np.dot(np.array([px, py]) - a, ab) /
                                    (np.dot(ab, ab) + 1e-12), 0, 1)
            proj = a + t * ab
            min_d = min(min_d, np.linalg.norm(np.array([px, py]) - proj))
        return min_d if self.contains(point) else -min_d

    def margin(self, point):
        """Alias for signed_distance (balance margin of a point)."""
        return self.signed_distance(point)
