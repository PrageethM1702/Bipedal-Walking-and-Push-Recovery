import numpy as np
from scipy.linalg import solve_discrete_are


class ZMPPreviewController:
    """
    Preview Controller for ZMP tracking (Kajita et al. ICRA 2003).
    """

    def __init__(self, z_c=0.852, g=9.81, dt=1/240, preview_steps=320):
        self.dt            = dt
        self.preview_steps = preview_steps
        self.z_c           = z_c
        self.g             = g

        self.A = np.array([
            [1, dt, dt**2/2],
            [0, 1,  dt     ],
            [0, 0,  1      ],
        ])
        self.b = np.array([dt**3/6, dt**2/2, dt])
        self.c = np.array([1, 0, -z_c/g])

        self.Q = np.eye(1) * 1.0
        self.R = np.eye(1) * 1e-6

        self._compute_gains()

        self.ei_x  = 0.0
        self.ei_y  = 0.0
        self.state_x = np.zeros(3)
        self.state_y = np.zeros(3)

    def _compute_gains(self):
        A = self.A
        b = self.b.reshape(3, 1)
        c = self.c.reshape(1, 3)

        Aa = np.block([
            [np.ones((1,1)),  c @ A         ],
            [np.zeros((3,1)), A             ],
        ])
        Ba = np.vstack([c @ b, b])

        Qa = np.zeros((4, 4))
        Qa[0, 0] = self.Q[0, 0]
        Ra = self.R

        try:
            P  = solve_discrete_are(Aa, Ba, Qa, Ra)
            K  = np.linalg.inv(Ra + Ba.T @ P @ Ba) @ Ba.T @ P @ Aa
            self.Gi = float(K[0, 0])
            self.Gx = K[0, 1:].flatten()
        except Exception as e:
            print(f"[ZMPPreview] DARE failed ({e}), using fallback gains")
            self.Gi = 1.0
            self.Gx = np.array([10.0, 2.0, 0.0])
            K = np.array([[self.Gi, *self.Gx]])

        Ac = Aa - Ba @ (np.linalg.inv(Ra + Ba.T @ P @ Ba) @ Ba.T @ P @ Aa)
        e0 = np.zeros((4, 1))
        e0[0, 0] = 1.0
        X  = -Ac.T @ P @ e0
        coeff = np.linalg.inv(Ra + Ba.T @ P @ Ba) @ Ba.T
        self.Gd = np.zeros(self.preview_steps)
        for i in range(self.preview_steps):
            self.Gd[i] = float((coeff @ X)[0, 0])
            X = Ac.T @ X

        print(f"[ZMPPreview] Gi={self.Gi:.4f}  Gx={self.Gx}  Gd[0]={self.Gd[0]:.6f}")

    def reset(self, x=0.0, y=0.0):
        self.state_x = np.array([x, 0.0, 0.0])
        self.state_y = np.array([y, 0.0, 0.0])
        self.ei_x    = 0.0
        self.ei_y    = 0.0

    def _step_axis(self, state, ei, zmp_ref_now, zmp_preview):
        zmp_current = float(self.c @ state)
        e      = zmp_current - zmp_ref_now
        ei_new = ei + e

        n_prev = min(len(zmp_preview), self.preview_steps)
        preview_sum = float(np.dot(self.Gd[:n_prev],
                                   zmp_preview[:n_prev]))

        u = -self.Gi * ei_new - float(self.Gx @ state) - preview_sum
        new_state = self.A @ state + self.b * u
        return new_state, ei_new

    def step(self, zmp_ref_x_now, zmp_ref_y_now,
             zmp_preview_x, zmp_preview_y):
        self.state_x, self.ei_x = self._step_axis(
            self.state_x, self.ei_x, zmp_ref_x_now, zmp_preview_x)
        self.state_y, self.ei_y = self._step_axis(
            self.state_y, self.ei_y, zmp_ref_y_now, zmp_preview_y)
        return (self.state_x[0], self.state_y[0],
                self.state_x[1], self.state_y[1])

    @property
    def com_x(self):  return self.state_x[0]
    @property
    def com_y(self):  return self.state_y[0]
    @property
    def com_xdot(self): return self.state_x[1]
    @property
    def com_ydot(self): return self.state_y[1]