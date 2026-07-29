"""
control/state_machine.py

Finite-state machine that arbitrates between walking, standing balance, and the
push-recovery strategies.  The transitions are driven by how far the capture
point has moved relative to the support polygon and by the CoM speed:

    STAND  ── capture point near edge ──►  ANKLE_RECOVER
    STAND  ── capture point outside    ──►  STEP_RECOVER
    WALK   ── large disturbance        ──►  STEP_RECOVER
    *      ── recovered & slow         ──►  STAND
    *      ── trunk fell over          ──►  FALLEN

The controller owns the actual motions; this class only decides the mode.
"""
from enum import Enum


class State(Enum):
    STAND         = "STAND"
    WALK          = "WALK"
    ANKLE_RECOVER = "ANKLE_RECOVER"
    HIP_RECOVER   = "HIP_RECOVER"
    STEP_RECOVER  = "STEP_RECOVER"
    FALLEN        = "FALLEN"


class BalanceStateMachine:
    def __init__(self,
                 step_margin=-0.005,
                 step_speed=0.28,
                 ankle_margin=0.020,
                 hip_speed=0.45,
                 recovered_speed=0.10,
                 fallen_tilt=1.0,
                 fallen_height=0.15,
                 walk_disturb_tilt=0.50,
                 walk_disturb_speed=1.30):
        self.state = State.STAND
        self.fallen_height = fallen_height
        self.walk_disturb_tilt = walk_disturb_tilt
        self.walk_disturb_speed = walk_disturb_speed
        self.step_margin = step_margin
        self.step_speed = step_speed
        self.ankle_margin = ankle_margin
        self.hip_speed = hip_speed
        self.recovered_speed = recovered_speed
        self.fallen_tilt = fallen_tilt

    def update(self, cp_margin, com_speed, tilt, com_z=1.0,
               walking=False, step_active=False):
        """
        cp_margin   : signed distance of capture point to support edge (+inside).
        com_speed   : |CoM horizontal velocity| (m/s).
        tilt        : max(|roll|, |pitch|) of the trunk (rad).
        com_z       : CoM height (m) -- used to detect a real collapse.
        walking     : is the walk gait currently commanded?
        step_active : is a recovery step in progress?
        Returns the (possibly new) State.
        """
        if com_z < self.fallen_height:
            self.state = State.FALLEN
            return self.state

        if step_active:
            self.state = State.STEP_RECOVER
            return self.state

        if tilt > self.fallen_tilt:
            self.state = State.FALLEN
            return self.state

        if walking:
            if tilt > self.walk_disturb_tilt or com_speed > self.walk_disturb_speed:
                self.state = State.STEP_RECOVER
            else:
                self.state = State.WALK
            return self.state

        if cp_margin < self.step_margin and com_speed > self.step_speed:
            self.state = State.STEP_RECOVER
        elif com_speed > self.hip_speed:
            self.state = State.HIP_RECOVER
        elif cp_margin < self.ankle_margin:
            self.state = State.ANKLE_RECOVER
        elif com_speed < self.recovered_speed:
            self.state = State.WALK if walking else State.STAND
        return self.state
