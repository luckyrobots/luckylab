"""Neural network models for skrl agents.

Note: Supports asymmetric actor-critic. See skrl_wrapper.py for details.
"""

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from luckylab.rl.config import ActorCriticCfg

ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: tuple[int, ...],
    activation: str,
) -> nn.Sequential:
    """Build MLP with given architecture."""
    act_cls = ACTIVATIONS[activation.lower()]
    layers: list[nn.Module] = []
    prev = in_dim
    for dim in hidden_dims:
        layers.extend([nn.Linear(prev, dim), act_cls()])
        prev = dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class GaussianActor(GaussianMixin, Model):
    """Stochastic policy for PPO/SAC."""

    def __init__(
        self,
        obs_space,
        act_space,
        device,
        cfg: ActorCriticCfg,
        num_policy_obs: int | None = None,
        squash_output: bool = False,
    ):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20, max_log_std=2
        )

        self._num_policy_obs = num_policy_obs or obs_space.shape[0]
        self._squash_output = squash_output
        self.net = build_mlp(
            self._num_policy_obs, act_space.shape[0], cfg.actor_hidden_dims, cfg.activation
        )
        self._noise_std_type = cfg.noise_std_type

        if cfg.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.full((act_space.shape[0],), cfg.init_noise_std).log()
            )
        else:
            self.std = nn.Parameter(
                torch.full((act_space.shape[0],), cfg.init_noise_std)
            )

    def compute(self, inputs, role=""):
        obs = inputs["states"][:, :self._num_policy_obs]
        mean = self.net(obs)

        if self._squash_output:
            mean = torch.tanh(mean)

        if self._noise_std_type == "log":
            log_std = self.log_std.expand_as(mean)
        else:
            log_std = self.std.clamp(min=1e-6).log().expand_as(mean)
        return mean, log_std, {}


class DeterministicActor(DeterministicMixin, Model):
    """Deterministic policy for TD3/DDPG."""

    def __init__(
        self,
        obs_space,
        act_space,
        device,
        cfg: ActorCriticCfg,
        num_policy_obs: int | None = None,
    ):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self._num_policy_obs = num_policy_obs or obs_space.shape[0]
        self.net = build_mlp(
            self._num_policy_obs, act_space.shape[0], cfg.actor_hidden_dims, cfg.activation
        )

    def compute(self, inputs, role=""):
        obs = inputs["states"][:, :self._num_policy_obs]
        return torch.tanh(self.net(obs)), {}


class Critic(DeterministicMixin, Model):
    """Value function (V) for PPO."""

    def __init__(self, obs_space, act_space, device, cfg: ActorCriticCfg):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self)
        self.net = build_mlp(obs_space.shape[0], 1, cfg.critic_hidden_dims, cfg.activation)

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), {}


class QCritic(DeterministicMixin, Model):
    """Q-function for SAC/TD3/DDPG."""

    def __init__(self, obs_space, act_space, device, cfg: ActorCriticCfg):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self)
        self.net = build_mlp(
            obs_space.shape[0] + act_space.shape[0], 1, cfg.critic_hidden_dims, cfg.activation
        )

    def compute(self, inputs, role=""):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}
