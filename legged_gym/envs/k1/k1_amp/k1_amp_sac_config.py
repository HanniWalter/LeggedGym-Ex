from legged_gym.envs.k1.k1_amp.k1_amp_config import K1AMPCfg, MOTION_FILES
from legged_gym.envs.base.template_cfgs import LeggedRobotAMPCfgSAC, LeggedRobotAMPCfgFastSAC
from legged_gym import SIMULATOR


class K1AMPSACEnvCfg(K1AMPCfg):
    """SAC-specific env config. Overrides reward scales without touching PPO config."""
    class rewards(K1AMPCfg.rewards):
        class scales(K1AMPCfg.rewards.scales):
            # SAC-specific reward tuning (Apr12)
            tracking_lin_vel = 1.5   # was 1.2 — SAC needs stronger velocity signal
            keep_balance = 2.0       # was 1.0 — biggest gap vs PPO (0.47 vs 0.93)
            orientation = -3.0       # was -2.0 — stronger anti-falling penalty


class K1AMPCfgSAC(LeggedRobotAMPCfgSAC):
    """K1 AMP train config with SAC backend. Env config is K1AMPCfg (passed separately)."""

    class policy(LeggedRobotAMPCfgSAC.policy):
        # Match PPO critic capacity; large obs (375-dim) needs bigger nets
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [1024, 512, 256]
        activation = 'relu'
        use_layer_norm = False

    class algorithm(LeggedRobotAMPCfgSAC.algorithm):
        amp_replay_buffer_size = K1AMPCfg.env.num_envs * 24 * 10
        disc_lr = 1e-4
        # Bigger batches for more stable gradients with 8192 envs
        batch_size = 1024
        # TUNING (Apr12): reduced from 4 to 2 — fewer gradient steps per env step
        # to reduce overfitting to stale replay data. 4x UTD with 8192 envs is too aggressive.
        utd_ratio = 2
        # TUNING (Apr12): increased from 0.002 to 0.005 — faster target net update
        # reduces Q-value lag when policy changes rapidly
        tau = 0.005
        # TUNING (Apr12): reduced from 1e-4 to 3e-5 — slower alpha updates prevent
        # the alpha oscillation (0.03→0.10→0.03) seen in Apr11 runs
        alpha_lr = 3e-5
        # TUNING (Apr12): reduced from 1.0 to 0.2 — start lower to avoid early
        # entropy-driven chaos. 1.0 caused Q-values to be entropy-dominated initially.
        init_alpha = 0.2
        # TUNING (Apr12): raised from -11.0 to -5.0 — less aggressive entropy target.
        # -11 = -0.5*22 pushes alpha too low too fast, then it oscillates trying to recover.
        # -5.0 keeps moderate exploration without dramatic alpha swings.
        target_entropy = -5.0
        # TUNING (Apr12): reduced from 0.99 to 0.98 — shorter horizon reduces
        # Q-value variance, makes it easier to learn stable balance behaviors.
        gamma = 0.98
        # Normalize rewards in SAC (running mean/std)
        normalize_rewards = True
        # Obs normalization ON for 375-dim observations
        normalize_observations = True
        # Use AdamW for weight decay regularization
        use_adamw = True
        # No policy delay — update actor every critic step
        policy_delay = 1
        n_step = 1

    class runner(LeggedRobotAMPCfgSAC.runner):
        # TUNING (Apr12): reduced from 2.0 to 1.5 — slightly lower AMP magnitude
        # so task rewards (balance, velocity) are more visible to Q-network.
        # rew_amp was ~20-23 which is healthy; 1.5 should give ~15-17 (still good).
        amp_reward_coef = 1.5
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = K1AMPCfg.env.num_envs * 24 * 10
        amp_discr_hidden_dims = [1024, 512]
        # TUNING (Apr12): increased from 0.65 to 0.75 (75% task / 25% AMP)
        # Formula: blended = (1 - lerp) * amp + lerp * task = 0.25*amp + 0.75*task
        # Rationale: AMP style is already learned well (rew_amp=20+), but balance
        # (0.47 vs PPO's 0.93) and velocity tracking (0.35 vs PPO's 0.83) are poor.
        # More task weight pushes toward stable walking over stylish falling.
        amp_task_reward_lerp = 0.75
        # TUNING (Apr12): increased from *3 to *5 — longer warmup fills buffer with
        # more diverse transitions before policy optimization begins.
        warmup_steps = K1AMPCfg.env.num_envs * 5
        # Buffer must hold many iterations: 8192 envs × 24 steps = 197K per iter
        replay_buffer_size = 2_000_000

        max_iterations = 20060
        save_interval = 200
        run_name = f'k1_amp_sac_{SIMULATOR}'
        experiment_name = 'k1_amp_sac'


class K1AMPCfgFastSAC(LeggedRobotAMPCfgFastSAC):
    """K1 AMP train config with FastSAC backend (high UTD ratio). Env config is K1AMPCfg."""

    class policy(LeggedRobotAMPCfgFastSAC.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [1024, 512, 256]

    class algorithm(LeggedRobotAMPCfgFastSAC.algorithm):
        amp_replay_buffer_size = K1AMPCfg.env.num_envs * 24 * 10
        disc_lr = 1e-4
        target_entropy = -11.0
        normalize_rewards = True

    class runner(LeggedRobotAMPCfgFastSAC.runner):
        amp_reward_coef = 1.0 * K1AMPCfg.control.dt
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = K1AMPCfg.env.num_envs * 24 * 10
        amp_discr_hidden_dims = [1024, 512]
        amp_task_reward_lerp = 0.5

        max_iterations = 20060
        save_interval = 200
        run_name = f'k1_amp_fastsac_{SIMULATOR}'
        experiment_name = 'k1_amp_fastsac'
