"""
simulation/sim_manager.py

Sets up the PyBullet world and loads the 12-DoF biped.
"""
import os
import time
import pybullet as p
import pybullet_data

from simulation.robot_interface import ROBOT_URDF, SPAWN_Z

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BACKGROUND = (0.086, 0.094, 0.110)
FLOOR_TEXTURE = os.path.join(_ROOT, "assets", "floor_grid.png")
FLOOR_TINT = [0.62, 0.65, 0.70, 1.0]
FLOOR_SPECULAR = [0.05, 0.05, 0.05]


class SimManager:
    def __init__(self, gui=True, timestep=1.0 / 2000, solver_iters=200):
        self.gui          = gui
        self.timestep     = timestep
        self.solver_iters = solver_iters
        self.robot_id     = None

    def init(self):
        if self.gui:
            p.connect(p.GUI, options=(
                f"--background_color_red={BACKGROUND[0]} "
                f"--background_color_green={BACKGROUND[1]} "
                f"--background_color_blue={BACKGROUND[2]}"))
            for flag in (p.COV_ENABLE_GUI,
                         p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                         p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                         p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
                p.configureDebugVisualizer(flag, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.1, cameraYaw=52, cameraPitch=-24,
                cameraTargetPosition=[0.2, 0, 0.16],
            )
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.timestep)
        p.setPhysicsEngineParameter(numSolverIterations=self.solver_iters)
        p.setGravity(0, 0, -9.8)
        self._load_floor()

        urdf_path = os.path.join(_ROOT, ROBOT_URDF)
        self.robot_id = p.loadURDF(
            urdf_path, [0, 0, SPAWN_Z],
            p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False,
        )
        print(f"[SimManager] 12-DoF biped loaded | id={self.robot_id} | "
              f"{1/self.timestep:.0f} Hz")
        return self.robot_id

    def _load_floor(self):
        """Load the ground plane and dress it with the dark grid texture.

        The texture is generated on first use, so a fresh clone does not need
        any binary asset committed to the repo."""
        self.plane_id = p.loadURDF("plane.urdf")
        try:
            if not os.path.exists(FLOOR_TEXTURE):
                from tools.make_floor import make_floor
                make_floor(FLOOR_TEXTURE)
            tex = p.loadTexture(FLOOR_TEXTURE)
            p.changeVisualShape(self.plane_id, -1, textureUniqueId=tex,
                                rgbaColor=FLOOR_TINT,
                                specularColor=FLOOR_SPECULAR)
        except Exception as exc:
            print(f"[SimManager] floor texture unavailable ({exc}); "
                  f"using default plane")
        return self.plane_id

    def step(self):
        p.stepSimulation()

    def step_realtime(self):
        p.stepSimulation()
        time.sleep(self.timestep)

    def is_connected(self):
        return p.isConnected()

    def follow_camera(self, target):
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2, cameraYaw=40, cameraPitch=-20,
                cameraTargetPosition=[float(target[0]), float(target[1]), 0.15],
            )

    def disconnect(self):
        if p.isConnected():
            p.disconnect()
