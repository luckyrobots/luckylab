"""Tests for EntityData observation parsing, properties, and reset."""

import torch

from luckylab.entity.data import EntityData, ObservationSchema


class TestObservationSchema:
    """Tests for ObservationSchema mapping."""

    def test_default_schema_has_core_mappings(self):
        schema = ObservationSchema.default()
        assert "base_lin_vel" in schema.mappings
        assert "base_ang_vel" in schema.mappings
        assert "base_quat" in schema.mappings
        assert "joint_pos" in schema.mappings
        assert "joint_vel" in schema.mappings

    def test_get_size_fixed(self):
        schema = ObservationSchema.default()
        assert schema.get_size("base_lin_vel", num_joints=12) == 3
        assert schema.get_size("base_quat", num_joints=12) == 4

    def test_get_size_joint_dependent(self):
        schema = ObservationSchema.default()
        assert schema.get_size("joint_pos", num_joints=12) == 12
        assert schema.get_size("joint_pos", num_joints=6) == 6

    def test_get_size_unknown_returns_zero(self):
        schema = ObservationSchema.default()
        assert schema.get_size("nonexistent", num_joints=12) == 0

    def test_get_property(self):
        schema = ObservationSchema.default()
        assert schema.get_property("base_lin_vel") == "root_link_lin_vel_b"
        assert schema.get_property("joint_pos") == "joint_pos"
        assert schema.get_property("nonexistent") is None

    def test_aliases_map_to_same_property(self):
        schema = ObservationSchema.default()
        assert schema.get_property("base_lin_vel") == schema.get_property("lin_vel")
        assert schema.get_property("base_ang_vel") == schema.get_property("ang_vel")


class TestEntityDataInit:
    """Tests for EntityData initialization."""

    def test_buffer_shapes(self):
        data = EntityData(num_envs=4, device=torch.device("cpu"), num_joints=12)
        assert data.root_link_lin_vel_b.shape == (4, 3)
        assert data.root_link_ang_vel_b.shape == (4, 3)
        assert data.root_link_quat_w.shape == (4, 4)
        assert data.projected_gravity_b.shape == (4, 3)
        assert data.joint_pos.shape == (4, 12)
        assert data.joint_vel.shape == (4, 12)

    def test_default_quaternion_is_identity(self):
        data = EntityData(num_envs=2, device=torch.device("cpu"), num_joints=6)
        # w=1, x=0, y=0, z=0
        assert data.root_link_quat_w[0, 0].item() == 1.0
        assert data.root_link_quat_w[0, 1:].sum().item() == 0.0

    def test_default_gravity_is_downward(self):
        data = EntityData(num_envs=2, device=torch.device("cpu"), num_joints=6)
        assert data.projected_gravity_b[0, 2].item() == -1.0

    def test_foot_buffers(self):
        data = EntityData(num_envs=2, device=torch.device("cpu"), num_joints=12)
        assert data.foot_contact.shape == (2, 4)
        assert data.foot_height.shape == (2, 4)
        assert data.foot_contact_forces.shape == (2, 4)
        assert data.foot_air_time.shape == (2, 4)


class TestEntityDataUpdateFromObservation:
    """Tests for update_from_observation parsing."""

    def test_parse_linear_velocity(self):
        data = EntityData(num_envs=1, device=torch.device("cpu"), num_joints=12)
        obs = torch.tensor([[1.0, 2.0, 3.0]])
        data.update_from_observation(obs, ["base_lin_vel"])
        assert torch.allclose(data.root_link_lin_vel_b, torch.tensor([[1.0, 2.0, 3.0]]))

    def test_parse_multiple_observations(self):
        data = EntityData(num_envs=1, device=torch.device("cpu"), num_joints=3)
        # lin_vel(3) + ang_vel(3) + joint_pos(3) = 9 values
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3]])
        data.update_from_observation(obs, ["base_lin_vel", "base_ang_vel", "joint_pos"])
        assert torch.allclose(data.root_link_lin_vel_b, torch.tensor([[1.0, 2.0, 3.0]]))
        assert torch.allclose(data.root_link_ang_vel_b, torch.tensor([[4.0, 5.0, 6.0]]))
        assert torch.allclose(data.joint_pos, torch.tensor([[0.1, 0.2, 0.3]]))

    def test_1d_obs_tensor_gets_unsqueezed(self):
        data = EntityData(num_envs=1, device=torch.device("cpu"), num_joints=12)
        obs = torch.tensor([1.0, 2.0, 3.0])  # 1D
        data.update_from_observation(obs, ["base_lin_vel"])
        assert torch.allclose(data.root_link_lin_vel_b, torch.tensor([[1.0, 2.0, 3.0]]))

    def test_unknown_observation_raises(self):
        import pytest

        data = EntityData(num_envs=1, device=torch.device("cpu"), num_joints=12)
        obs = torch.zeros(1, 5)
        with pytest.raises(ValueError, match="Unknown observation"):
            data.update_from_observation(obs, ["nonexistent_obs"])

    def test_quaternion_parsing(self):
        data = EntityData(num_envs=1, device=torch.device("cpu"), num_joints=12)
        obs = torch.tensor([[0.707, 0.0, 0.707, 0.0]])
        data.update_from_observation(obs, ["base_quat"])
        assert torch.allclose(
            data.root_link_quat_w, torch.tensor([[0.707, 0.0, 0.707, 0.0]]), atol=1e-3
        )


class TestEntityDataReset:
    """Tests for EntityData.reset()."""

    def test_reset_zeroes_velocities(self):
        data = EntityData(num_envs=2, device=torch.device("cpu"), num_joints=6)
        data._root_link_lin_vel_b[:] = 5.0
        data._root_link_ang_vel_b[:] = 3.0
        data.reset(torch.tensor([0]))
        assert (data.root_link_lin_vel_b[0] == 0.0).all()
        assert (data.root_link_ang_vel_b[0] == 0.0).all()
        # env 1 untouched
        assert (data.root_link_lin_vel_b[1] == 5.0).all()

    def test_reset_restores_identity_quaternion(self):
        data = EntityData(num_envs=2, device=torch.device("cpu"), num_joints=6)
        data._root_link_quat_w[:] = 0.0
        data.reset(torch.tensor([0, 1]))
        assert data.root_link_quat_w[0, 0].item() == 1.0
        assert data.root_link_quat_w[0, 1:].sum().item() == 0.0

    def test_reset_restores_default_joint_pos(self):
        data = EntityData(num_envs=2, device=torch.device("cpu"), num_joints=3)
        data.set_default_joint_pos([0.1, 0.2, 0.3])
        data._joint_pos[:] = 99.0
        data.reset(torch.tensor([0]))
        assert torch.allclose(data.joint_pos[0], torch.tensor([0.1, 0.2, 0.3]))
        assert (data.joint_pos[1] == 99.0).all()


class TestEntityDataDerivedProperties:
    """Tests for derived properties like heading and world-frame velocities."""

    def test_heading_zero_when_identity_quat(self):
        data = EntityData(num_envs=1, device=torch.device("cpu"), num_joints=6)
        # Identity quaternion -> forward is +x -> heading = atan2(0,1) = 0
        heading = data.heading_w
        assert abs(heading[0].item()) < 1e-5

    def test_world_velocity_equals_body_for_identity_quat(self):
        data = EntityData(num_envs=1, device=torch.device("cpu"), num_joints=6)
        data._root_link_lin_vel_b[0] = torch.tensor([1.0, 2.0, 3.0])
        vel_w = data.root_link_lin_vel_w
        assert torch.allclose(vel_w, data.root_link_lin_vel_b, atol=1e-5)

    def test_aliases(self):
        data = EntityData(num_envs=1, device=torch.device("cpu"), num_joints=6)
        assert data.base_lin_vel is data.root_link_lin_vel_b
        assert data.base_ang_vel is data.root_link_ang_vel_b
        assert data.projected_gravity is data.projected_gravity_b


class TestEntityDataJointLimits:
    """Tests for joint limit and action scale configuration."""

    def test_set_joint_pos_limits(self):
        data = EntityData(num_envs=2, device=torch.device("cpu"), num_joints=3)
        data.set_joint_pos_limits(
            lower=[-1.0, -2.0, -3.0],
            upper=[1.0, 2.0, 3.0],
        )
        assert data.soft_joint_pos_limits.shape == (2, 3, 2)
        assert data.soft_joint_pos_limits[0, 0, 0].item() == -1.0
        assert data.soft_joint_pos_limits[0, 0, 1].item() == 1.0

    def test_set_action_scale(self):
        data = EntityData(num_envs=1, device=torch.device("cpu"), num_joints=3)
        data.set_action_scale([0.5, 1.0, 1.5])
        assert torch.allclose(data.action_scale, torch.tensor([0.5, 1.0, 1.5]))

    def test_default_action_scale_is_ones(self):
        data = EntityData(num_envs=1, device=torch.device("cpu"), num_joints=3)
        assert (data.action_scale == 1.0).all()
