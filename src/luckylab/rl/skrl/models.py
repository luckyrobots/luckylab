"""Neural network models for skrl agents."""

import math

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
    """Stochastic policy for PPO/SAC.

    When squash_output=True (SAC), overrides act() with proper reparameterized
    sampling: action = tanh(mean + std * eps), log_prob includes Jacobian.
    When squash_output=False (PPO), uses GaussianMixin's default sampling.
    """

    def __init__(
        self,
        obs_space,
        act_space,
        device,
        cfg: ActorCriticCfg,
        squash_output: bool = False,
    ):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(
            self, clip_actions=not squash_output, clip_log_std=True,
            min_log_std=-20, max_log_std=2,
        )

        self._squash_output = squash_output
        self._epsilon = 1e-6
        self.net = build_mlp(
            obs_space.shape[0], act_space.shape[0], cfg.actor_hidden_dims, cfg.activation
        )
        # Zero-init last layer so initial policy output is 0 (pure CPG)
        if squash_output:
            last_layer = self.net[-1]
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)
        self.log_std = nn.Parameter(
            torch.full((act_space.shape[0],), cfg.init_noise_std).log()
        )

    def compute(self, inputs, role=""):
        """PPO path: return mean and log_std for GaussianMixin sampling."""
        mean = self.net(inputs["states"])
        log_std = torch.clamp(self.log_std, -20.0, 2.0).expand_as(mean)
        return mean, log_std, {}

    def act(self, inputs, role=""):
        """SAC path: reparameterized sampling with tanh squashing + Jacobian.

        PPO falls through to GaussianMixin.act() which calls compute().
        """
        if not self._squash_output:
            return super().act(inputs, role)
        mean = self.net(inputs["states"])
        log_std = torch.clamp(self.log_std, -20.0, 2.0).expand_as(mean)
        std = log_std.exp()

        eps = torch.randn_like(mean)
        z = mean + std * eps
        actions = torch.tanh(z)

        # log p(a|s) = log N(z; mean, std) - sum log(1 - tanh(z)^2)
        log_prob = -0.5 * (eps.pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.sum(
            torch.log(1.0 - actions.pow(2) + self._epsilon), dim=-1, keepdim=True
        )

        return actions, log_prob, {"mean_actions": torch.tanh(mean)}


class GSDEActor(GaussianMixin, Model):
    """SAC actor with gSDE. Matches SB3's StateDependentNoiseDistribution.

    Noise = features @ exploration_matrix (resampled every N steps).
    Actions = tanh(mean + noise), log_prob includes tanh Jacobian correction.
    """

    def __init__(self, obs_space, act_space, device, cfg: ActorCriticCfg):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(
            self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2
        )

        n_obs = obs_space.shape[0]
        n_actions = act_space.shape[0]
        act_cls = ACTIVATIONS[cfg.activation.lower()]

        layers: list[nn.Module] = []
        prev = n_obs
        for dim in cfg.actor_hidden_dims:
            layers.extend([nn.Linear(prev, dim), act_cls()])
            prev = dim
        self.features_net = nn.Sequential(*layers)
        self.latent_dim = prev

        action_linear = nn.Linear(self.latent_dim, n_actions)
        nn.init.zeros_(action_linear.weight)
        nn.init.zeros_(action_linear.bias)
        self.action_head = nn.Sequential(
            action_linear,
            nn.Hardtanh(min_val=-2.0, max_val=2.0),
        )

        self.log_std = nn.Parameter(
            torch.full((self.latent_dim, n_actions), math.log(cfg.init_noise_std))
        )

        self.register_buffer(
            "_exploration_matrix",
            torch.zeros(self.latent_dim, n_actions),
            persistent=False,
        )
        self._step_counter = 0
        self._resample_interval = cfg.gsde_resample_interval
        self._epsilon = 1e-6
        self._sample_exploration()

    @torch.no_grad()
    def _sample_exploration(self) -> None:
        std = torch.exp(self.log_std)
        self._exploration_matrix.copy_(torch.randn_like(std) * std)

    def compute(self, inputs, role=""):
        raise NotImplementedError("Use act() instead.")

    def act(self, inputs, role=""):
        # Only resample exploration during env interaction (eval mode),
        # not during gradient updates (train mode). skrl sets train mode
        # in _update() and eval mode for env interaction.
        if not self.training:
            self._step_counter += 1
            if self._step_counter % self._resample_interval == 0:
                self._sample_exploration()

        features = self.features_net(inputs["states"])
        mean_raw = self.action_head(features)

        # Detach features for noise so gradients only flow through the mean
        # action path (action_head), not through the random exploration_matrix.
        # Without detach, features_net receives random gradients via
        # exploration_matrix^T that globally shift noise patterns for all states.
        features_detached = features.detach()
        noise = features_detached @ self._exploration_matrix
        pre_squash = mean_raw + noise
        actions = torch.tanh(pre_squash)

        # Log-prob with marginal variance and tanh Jacobian correction
        std_matrix = torch.exp(self.log_std)
        std = torch.sqrt(features_detached.pow(2) @ std_matrix.pow(2) + self._epsilon)

        log_prob = -0.5 * (
            ((pre_squash - mean_raw) / std).pow(2)
            + 2.0 * torch.log(std)
            + math.log(2.0 * math.pi)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.sum(
            torch.log(1.0 - actions.pow(2) + self._epsilon), dim=-1, keepdim=True
        )

        return actions, log_prob, {"mean_actions": torch.tanh(mean_raw)}


class DeterministicActor(DeterministicMixin, Model):
    """Deterministic policy for TD3/DDPG."""

    def __init__(
        self,
        obs_space,
        act_space,
        device,
        cfg: ActorCriticCfg,
    ):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = build_mlp(
            obs_space.shape[0], act_space.shape[0], cfg.actor_hidden_dims, cfg.activation
        )

    def compute(self, inputs, role=""):
        return torch.tanh(self.net(inputs["states"])), {}


class Critic(DeterministicMixin, Model):
    """Value function (V) for PPO."""

    def __init__(self, obs_space, act_space, device, cfg: ActorCriticCfg):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self)
        self.net = build_mlp(obs_space.shape[0], 1, cfg.critic_hidden_dims, cfg.activation)

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), {}


class QCritic(DeterministicMixin, Model):
    """Q-function for SAC/TD3/DDPG. Uses LayerNorm for activation stability."""

    def __init__(self, obs_space, act_space, device, cfg: ActorCriticCfg):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self)
        in_dim = obs_space.shape[0] + act_space.shape[0]
        act_cls = ACTIVATIONS[cfg.activation.lower()]
        layers: list[nn.Module] = []
        prev = in_dim
        for dim in cfg.critic_hidden_dims:
            layers.extend([nn.Linear(prev, dim), nn.LayerNorm(dim), act_cls()])
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role=""):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}
