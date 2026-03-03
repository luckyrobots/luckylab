"""Training utilities for skrl backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from skrl.resources.preprocessors.torch import RunningStandardScaler

from luckylab.rl.common import make_experiment_name, print_config, wrap_env
from luckylab.rl.config import RlRunnerCfg
from luckylab.rl.skrl.models import Critic, DeterministicActor, GaussianActor, GSDEActor, QCritic
from luckylab.utils.logging import WandbLogger, print_info
from luckylab.utils.random import seed_rng

if TYPE_CHECKING:
    from luckylab.envs import ManagerBasedRlEnvCfg
    from luckylab.rl.skrl.wrapper import SkrlWrapper


def _wrap_env(env, rl_cfg: RlRunnerCfg) -> SkrlWrapper:
    from luckylab.rl.skrl.wrapper import SkrlWrapper
    return wrap_env(env, rl_cfg, SkrlWrapper)


def create_agent(env, cfg: RlRunnerCfg, device: str, experiment_name: str | None = None):
    """Create skrl agent for the selected algorithm."""
    from skrl.memories.torch import RandomMemory

    obs_space, act_space = env.observation_space, env.action_space

    experiment_cfg = {
        "directory": cfg.directory,
        "experiment_name": experiment_name or cfg.experiment_name,
        "write_interval": cfg.log_interval,
        "checkpoint_interval": cfg.checkpoint_interval,
        "wandb": False,
    }

    if cfg.algorithm == "ppo":
        from skrl.agents.torch.ppo import PPO

        ppo = cfg.ppo
        models = {
            "policy": GaussianActor(obs_space, act_space, device, cfg.policy),
            "value": Critic(obs_space, act_space, device, cfg.policy),
        }
        memory_size = ppo.num_steps_per_env
        agent_cfg = {
            "rollouts": ppo.num_steps_per_env,
            "learning_epochs": ppo.num_learning_epochs,
            "mini_batches": ppo.num_mini_batches,
            "discount_factor": ppo.gamma,
            "lambda": ppo.lam,
            "learning_rate": ppo.learning_rate,
            "ratio_clip": ppo.clip_param,
            "value_clip": ppo.clip_param,
            "clip_predicted_values": ppo.use_clipped_value_loss,
            "value_loss_scale": ppo.value_loss_coef,
            "entropy_loss_scale": ppo.entropy_coef,
            "grad_norm_clip": ppo.max_grad_norm,
            "kl_threshold": ppo.desired_kl,
        }
        if ppo.schedule == "adaptive":
            from skrl.resources.schedulers.torch import KLAdaptiveLR
            agent_cfg["learning_rate_scheduler"] = KLAdaptiveLR
            agent_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": ppo.desired_kl}
        agent_cls = PPO

    elif cfg.algorithm == "sac":
        from skrl.agents.torch.sac import SAC

        sac = cfg.sac
        if cfg.policy.noise_type == "gsde":
            policy = GSDEActor(obs_space, act_space, device, cfg.policy)
        else:
            policy = GaussianActor(obs_space, act_space, device, cfg.policy, squash_output=True)
        models = {
            "policy": policy,
            "critic_1": QCritic(obs_space, act_space, device, cfg.policy),
            "critic_2": QCritic(obs_space, act_space, device, cfg.policy),
            "target_critic_1": QCritic(obs_space, act_space, device, cfg.policy),
            "target_critic_2": QCritic(obs_space, act_space, device, cfg.policy),
        }
        memory_size = sac.memory_size

        # The reward manager scales rewards by dt for decimation-invariance
        # (shared with PPO and other algorithms). SAC's entropy term doesn't
        # scale with dt, so we undo it here before rewards enter the replay
        # buffer. This keeps reward weights algorithm-agnostic.
        step_dt = env.unwrapped.step_dt
        agent_cfg = {
            "batch_size": sac.batch_size,
            "discount_factor": sac.discount_factor,
            "polyak": sac.polyak,
            "actor_learning_rate": sac.actor_learning_rate,
            "critic_learning_rate": sac.critic_learning_rate,
            "learn_entropy": sac.learn_entropy,
            "entropy_learning_rate": sac.actor_learning_rate,
            "initial_entropy_value": sac.initial_entropy,
            "grad_norm_clip": sac.grad_norm_clip,
            "random_timesteps": sac.random_timesteps,
            "learning_starts": sac.learning_starts,
            "gradient_steps": sac.gradient_steps,
            "rewards_shaper": lambda rewards, *args, step_dt=step_dt, **kwargs: rewards / step_dt,
            "state_preprocessor": RunningStandardScaler,
            "state_preprocessor_kwargs": {"size": obs_space, "device": device},
        }
        if sac.target_entropy is not None:
            agent_cfg["target_entropy"] = sac.target_entropy
        if sac.lr_scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            agent_cfg["learning_rate_scheduler"] = CosineAnnealingLR
            agent_cfg["learning_rate_scheduler_kwargs"] = {"T_max": cfg.max_iterations, "eta_min": sac.lr_min}
        elif sac.lr_scheduler == "linear":
            from torch.optim.lr_scheduler import LinearLR
            agent_cfg["learning_rate_scheduler"] = LinearLR
            agent_cfg["learning_rate_scheduler_kwargs"] = {
                "start_factor": 1.0,
                "end_factor": sac.lr_min / sac.actor_learning_rate,
                "total_iters": cfg.max_iterations,
            }
        agent_cls = SAC

    elif cfg.algorithm == "td3":
        from skrl.agents.torch.td3 import TD3

        td3 = cfg.td3
        models = {
            "policy": DeterministicActor(obs_space, act_space, device, cfg.policy),
            "target_policy": DeterministicActor(obs_space, act_space, device, cfg.policy),
            "critic_1": QCritic(obs_space, act_space, device, cfg.policy),
            "critic_2": QCritic(obs_space, act_space, device, cfg.policy),
            "target_critic_1": QCritic(obs_space, act_space, device, cfg.policy),
            "target_critic_2": QCritic(obs_space, act_space, device, cfg.policy),
        }
        memory_size = td3.memory_size
        agent_cfg = {
            "batch_size": td3.batch_size,
            "discount_factor": td3.discount_factor,
            "polyak": td3.polyak,
            "actor_learning_rate": td3.actor_learning_rate,
            "critic_learning_rate": td3.critic_learning_rate,
            "policy_delay": td3.policy_delay,
            "smooth_regularization_noise": td3.smooth_regularization_noise,
            "smooth_regularization_clip": td3.smooth_regularization_clip,
        }
        agent_cls = TD3

    elif cfg.algorithm == "ddpg":
        from skrl.agents.torch.ddpg import DDPG

        ddpg = cfg.ddpg
        models = {
            "policy": DeterministicActor(obs_space, act_space, device, cfg.policy),
            "target_policy": DeterministicActor(obs_space, act_space, device, cfg.policy),
            "critic": QCritic(obs_space, act_space, device, cfg.policy),
            "target_critic": QCritic(obs_space, act_space, device, cfg.policy),
        }
        memory_size = ddpg.memory_size
        agent_cfg = {
            "batch_size": ddpg.batch_size,
            "discount_factor": ddpg.discount_factor,
            "polyak": ddpg.polyak,
            "actor_learning_rate": ddpg.actor_learning_rate,
            "critic_learning_rate": ddpg.critic_learning_rate,
        }
        agent_cls = DDPG

    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

    agent_cfg["experiment"] = experiment_cfg
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)
    for m in models.values():
        m.to(device)

    agent = agent_cls(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )

    return agent


def train(env_cfg: ManagerBasedRlEnvCfg, rl_cfg: RlRunnerCfg, device: str = "cpu") -> None:
    """Train an RL agent using skrl."""
    from skrl.trainers.torch import SequentialTrainer

    from luckylab.envs import ManagerBasedRlEnv

    seed_rng(rl_cfg.seed)
    experiment_name = make_experiment_name(rl_cfg)
    Path(rl_cfg.directory, experiment_name).mkdir(parents=True, exist_ok=True)

    with WandbLogger(rl_cfg, experiment_name) as wb:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

        if rl_cfg.rerun:
            from luckylab.utils.rerun_logger import RerunLogger
            env.rerun_logger = RerunLogger(
                app_id=f"luckylab/{experiment_name}",
                save_path=rl_cfg.rerun_save_path,
                log_interval=rl_cfg.rerun_log_interval,
                env_idx=rl_cfg.rerun_env_idx,
            )

        print_config(env, rl_cfg, experiment_name, device)
        wrapped = _wrap_env(env, rl_cfg)
        agent = create_agent(wrapped, rl_cfg, device, experiment_name)

        if rl_cfg.resume_from:
            checkpoint_path = Path(rl_cfg.resume_from)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            agent.load(str(checkpoint_path))
            print_info(f"Resumed from checkpoint: {checkpoint_path}")

        wb.attach(agent, env=wrapped)

        if rl_cfg.algorithm == "ppo":
            timesteps = rl_cfg.max_iterations * rl_cfg.ppo.num_steps_per_env
        else:
            timesteps = rl_cfg.max_iterations

        trainer = SequentialTrainer(
            cfg={"timesteps": timesteps, "headless": True, "environment_info": "episode"},
            env=wrapped,
            agents=agent,
        )
        trainer.train()

        if env.rerun_logger is not None:
            env.rerun_logger.close()
        wrapped.close()


def load_agent(
    checkpoint_path: str,
    env_cfg: ManagerBasedRlEnvCfg,
    rl_cfg: RlRunnerCfg,
    device: str = "cpu",
):
    """Load a trained agent from checkpoint."""
    from luckylab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = _wrap_env(env, rl_cfg)
    agent = create_agent(wrapped, rl_cfg, device)
    agent.load(checkpoint_path)
    return agent, wrapped
