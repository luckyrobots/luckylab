"""Observation parsing utilities for velocity task."""

import numpy as np


class ObservationParser:
    """Parser for flat observation arrays from LuckyRobots.

    Converts flat numpy arrays into structured dictionaries with named components.
    """

    def __init__(self, num_joints: int) -> None:
        """
        Initialize observation parser.

        Args:
            num_joints: Number of joints in the robot
        """
        self.num_joints = num_joints
        # Observation structure:
        # [0:4] = commands: [vx, vy, wz, heading]
        # [4:7] = base_lin_vel: [x, y, z] (body frame)
        # [7:10] = base_ang_vel: [x, y, z] (body frame)
        # [10:10+num_joints] = joint_pos (relative to default)
        # [10+num_joints:10+2*num_joints] = joint_vel
        # [10+2*num_joints:10+3*num_joints] = last_act
        self.commands_idx = slice(0, 4)
        self.base_lin_vel_idx = slice(4, 7)
        self.base_ang_vel_idx = slice(7, 10)
        self.joint_pos_idx = slice(10, 10 + num_joints)
        self.joint_vel_idx = slice(10 + num_joints, 10 + 2 * num_joints)
        self.last_act_idx = slice(10 + 2 * num_joints, 10 + 3 * num_joints)

    def parse(self, obs_array: np.ndarray) -> dict[str, np.ndarray]:
        """
        Parse flat observation array into components.

        Args:
            obs_array: Flat observation array

        Returns:
            Dictionary with parsed components
        """
        return {
            "commands": obs_array[self.commands_idx],  # [vx, vy, wz, heading]
            "base_lin_vel": obs_array[self.base_lin_vel_idx],  # [x, y, z]
            "base_ang_vel": obs_array[self.base_ang_vel_idx],  # [x, y, z]
            "joint_pos": obs_array[self.joint_pos_idx],  # relative to default
            "joint_vel": obs_array[self.joint_vel_idx],
            "last_act": obs_array[self.last_act_idx],
        }
