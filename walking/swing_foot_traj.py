import numpy as np


class SwingFootTrajectory:
    """
    Generates a smooth swing foot trajectory using cubic spline interpolation.
    Foot lifts to step_height at midpoint, lands at target position.
    """

    def __init__(self, step_height=0.06, dt=1/240):
        self.step_height = step_height
        self.dt          = dt

    def generate(self, start_pos, end_pos, duration):
        """
        start_pos: [x, y, z] of foot at lift-off
        end_pos:   [x, y, z] of foot at landing
        duration:  seconds for the swing

        Returns array of (x, y, z) positions per timestep.
        """
        n = int(duration / self.dt)
        traj = np.zeros((n, 3))

        for i in range(n):
            t = i / max(n - 1, 1)

            t_smooth = t * t * (3 - 2 * t)

            x = start_pos[0] + t_smooth * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t_smooth * (end_pos[1] - start_pos[1])

            z_base = start_pos[2] + t_smooth * (end_pos[2] - start_pos[2])
            z_arc  = self.step_height * 4 * t * (1 - t)
            z      = z_base + z_arc

            traj[i] = [x, y, z]

        return traj