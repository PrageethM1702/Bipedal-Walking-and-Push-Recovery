"""
simulation/robot_interface.py

Thin wrapper around the PyBullet body of the 12-DoF bipedal robot
(`robot/humanoid_leg_12dof.urdf`).  It exposes:

  * the joint / link index maps,
  * sensor reads (base pose, velocity, CoM, foot contacts, foot poses),
  * a position-controlled ``apply_motor_angles`` actuator,
  * ``apply_external_force`` used by the push applicator.

The robot is a light (~1.7 kg) leg-only biped.  Each leg is a 6-DoF chain:

    hip_yaw(Z) -> hip_roll(X) -> hip_pitch(Y) -> knee(Y) -> ankle_pitch(Y) -> ankle_roll(X)

The 12 motors are ordered *right leg first, then left leg*, matching the order
the inverse-kinematics module emits, so an IK vector can be sent straight to
``apply_motor_angles``.
"""
import numpy as np
import pybullet as p

ROBOT_URDF = "robot/humanoid_leg_12dof.urdf"
SPAWN_Z    = 0.31

R_HIP_Z, R_HIP_X, R_HIP_Y, R_KNEE, R_ANK_Y, R_ANK_X = 1, 2, 3, 4, 5, 6
L_HIP_Z, L_HIP_X, L_HIP_Y, L_KNEE, L_ANK_Y, L_ANK_X = 17, 18, 19, 20, 21, 22

RIGHT_LEG_JOINTS = [R_HIP_Z, R_HIP_X, R_HIP_Y, R_KNEE, R_ANK_Y, R_ANK_X]
LEFT_LEG_JOINTS  = [L_HIP_Z, L_HIP_X, L_HIP_Y, L_KNEE, L_ANK_Y, L_ANK_X]
MOTOR_JOINTS     = RIGHT_LEG_JOINTS + LEFT_LEG_JOINTS

IDX_ANKLE_PITCH = 4
IDX_ANKLE_ROLL  = 5
IDX_HIP_ROLL    = 1
IDX_HIP_PITCH   = 2

FOOT_LINK_R = 8
FOOT_LINK_L = 24
FOOT_LINKS_R = {7, 8}
FOOT_LINKS_L = {23, 24}

FOOT_HALF_LEN = 0.05
FOOT_HALF_WID = 0.026

MOTOR_KP       = 0.5
MOTOR_KD       = 0.5
MOTOR_TORQUE   = 1.5
MOTOR_MAX_VEL  = 5.0


class RobotInterface:
    def __init__(self, robot_id):
        self.robot_id = robot_id

    def get_base_position(self):
        return np.array(p.getBasePositionAndOrientation(self.robot_id)[0])

    def get_base_orientation(self):
        orn = p.getBasePositionAndOrientation(self.robot_id)[1]
        return np.array(p.getEulerFromQuaternion(orn))

    def get_base_velocity(self):
        lin, ang = p.getBaseVelocity(self.robot_id)
        return np.array(lin), np.array(ang)

    def get_com_position(self):
        """Whole-body centre of mass in world coordinates."""
        total_m = p.getDynamicsInfo(self.robot_id, -1)[0]
        pos = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        weighted = total_m * pos
        for j in range(p.getNumJoints(self.robot_id)):
            m = p.getDynamicsInfo(self.robot_id, j)[0]
            if m <= 0.0:
                continue
            link_pos = np.array(p.getLinkState(self.robot_id, j)[0])
            weighted += m * link_pos
            total_m += m
        return weighted / total_m

    def get_com_velocity(self):
        """Approximate CoM velocity by the base (trunk) linear velocity.

        The trunk holds ~half the mass and the legs are light, so the base
        velocity is a good, cheap proxy for the CoM velocity."""
        return self.get_base_velocity()[0]

    def get_torso_tilt(self):
        rpy = self.get_base_orientation()
        return rpy[0], rpy[1]

    def get_link_world_position(self, link_index):
        return np.array(p.getLinkState(self.robot_id, link_index)[0])

    def get_left_foot_pos(self):
        return self.get_link_world_position(FOOT_LINK_L)

    def get_right_foot_pos(self):
        return self.get_link_world_position(FOOT_LINK_R)

    def get_foot_contacts(self):
        """Return (left_in_contact, right_in_contact) booleans."""
        left = right = False
        for c in p.getContactPoints(bodyA=self.robot_id):
            link = c[3]
            if link in FOOT_LINKS_L:
                left = True
            if link in FOOT_LINKS_R:
                right = True
        return left, right

    def get_motor_angles(self):
        states = p.getJointStates(self.robot_id, MOTOR_JOINTS)
        return np.array([s[0] for s in states])

    def get_motor_velocities(self):
        states = p.getJointStates(self.robot_id, MOTOR_JOINTS)
        return np.array([s[1] for s in states])

    def apply_motor_angles(self, angles, kp=MOTOR_KP, kd=MOTOR_KD,
                           torque=MOTOR_TORQUE, max_velocity=MOTOR_MAX_VEL):
        """Position-control the 12 motors.  ``angles`` is length-12 in
        MOTOR_JOINTS order (right leg, then left leg)."""
        for idx, angle in zip(MOTOR_JOINTS, angles):
            p.setJointMotorControl2(
                self.robot_id, idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(angle),
                positionGain=kp,
                velocityGain=kd,
                force=torque,
                maxVelocity=max_velocity,
            )

    def reset_motor_angles(self, angles):
        """Teleport the joints (used once at start-up)."""
        for idx, angle in zip(MOTOR_JOINTS, angles):
            p.resetJointState(self.robot_id, idx, float(angle), 0.0)

    def apply_external_force(self, force_vec, link_index=-1, world_frame=True):
        """Apply a force (N) to a link (default: the base/trunk)."""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        p.applyExternalForce(
            self.robot_id, link_index,
            forceObj=list(force_vec),
            posObj=list(pos),
            flags=p.WORLD_FRAME if world_frame else p.LINK_FRAME,
        )
