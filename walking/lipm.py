import numpy as np


class LIPM:
    """
    Linear Inverted Pendulum Model (LIPM)
    Models the robot as a point mass on a massless leg at fixed CoM height.
    State: [x, x_dot, y, y_dot]
    ZMP:   [px, py]

    From Kajita et al. ICRA 2003:
      x_ddot = (g / z_c) * (x - px)
      y_ddot = (g / z_c) * (y - py)
    """

    def __init__(self, z_c=0.852, g=9.81, dt=1/240):
        self.z_c  = z_c
        self.g    = g
        self.dt   = dt
        self.omega = np.sqrt(g / z_c)

        self.state = np.zeros(4)

        ch = np.cosh(self.omega * dt)
        sh = np.sinh(self.omega * dt)
        w  = self.omega

        self.A = np.array([
            [ch,    sh/w,  0,    0   ],
            [w*sh,  ch,    0,    0   ],
            [0,     0,     ch,   sh/w],
            [0,     0,     w*sh, ch  ],
        ])
        self.B = np.array([
            [1 - ch  ],
            [-w * sh  ],
            [0        ],
            [0        ],
        ])
        self.Bx = np.array([1-ch, -w*sh, 0,    0   ])
        self.By = np.array([0,    0,     1-ch, -w*sh])

    def reset(self, x=0.0, xdot=0.0, y=0.0, ydot=0.0):
        self.state = np.array([x, xdot, y, ydot])

    def step(self, zmp_x, zmp_y):
        """Advance one timestep given ZMP reference."""
        s = self.state
        new_state = self.A @ s + self.Bx * zmp_x + self.By * zmp_y
        self.state = new_state
        return self.state.copy()

    @property
    def x(self):     return self.state[0]
    @property
    def xdot(self):  return self.state[1]
    @property
    def y(self):     return self.state[2]
    @property
    def ydot(self):  return self.state[3]

    def capture_point(self):
        """
        Capture Point (CP) — where robot must step to stop falling.
        CP = CoM_pos + CoM_vel / omega
        """
        cp_x = self.x + self.xdot / self.omega
        cp_y = self.y + self.ydot / self.omega
        return np.array([cp_x, cp_y])

    def divergent_component(self):
        """Divergent Component of Motion (DCM) — same as capture point."""
        return self.capture_point()