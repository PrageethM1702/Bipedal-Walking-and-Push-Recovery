import numpy as np


FOOT_SEPARATION = 0.18

class FootstepPlanner:
    """
    Generates a sequence of left/right footstep positions for straight walking.
    Each footstep is: (x, y, side)  where side = 'left' or 'right'
    """

    def __init__(self,
                 step_length=0.15,
                 step_width=0.18,
                 n_steps=20):
        self.step_length = step_length
        self.step_width  = step_width
        self.n_steps     = n_steps

    def generate(self, start_x=0.0, start_y=0.0, start_side='right'):
        """
        Returns list of (x, y, side) tuples.
        First step is a half-step, then full steps alternate left/right.
        """
        steps = []
        x = start_x
        side = start_side

        for i in range(self.n_steps):
            y_offset = -self.step_width/2 if side == 'right' else self.step_width/2

            sl = self.step_length/2 if i == 0 else self.step_length
            x += sl

            steps.append((x, start_y + y_offset, side))
            side = 'left' if side == 'right' else 'right'

        return steps

    def zmp_reference_for_step(self, current_foot, next_foot, step_duration, dt):
        """
        Generate ZMP reference trajectory for one step.
        ZMP stays under current support foot during swing,
        then transitions to next foot at end.
        Returns list of (zmp_x, zmp_y) per timestep.
        """
        n = int(step_duration / dt)
        zmp = []

        support_ratio = 0.85
        n_support = int(n * support_ratio)
        n_transfer = n - n_support

        for _ in range(n_support):
            zmp.append((current_foot[0], current_foot[1]))

        for i in range(n_transfer):
            t = i / max(n_transfer - 1, 1)
            zx = current_foot[0] + t * (next_foot[0] - current_foot[0])
            zy = current_foot[1] + t * (next_foot[1] - current_foot[1])
            zmp.append((zx, zy))

        return zmp