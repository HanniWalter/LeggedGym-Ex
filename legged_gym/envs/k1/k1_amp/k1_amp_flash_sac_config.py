from legged_gym import SIMULATOR
from legged_gym.envs.base.template_cfgs import LeggedRobotAMPCfgFlashSAC
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.k1.k1_amp.k1_amp_config import K1AMPCfg, MOTION_FILES
from legged_gym.envs.k1.k1_amp.k1_amp_sac_config import K1AMPSACEnvCfg


class K1AMPCfgFlashSAC(LeggedRobotAMPCfgFlashSAC):
    class env(LeggedRobotCfg.env):
        num_envs = 1024  # Override base K1 default of 8192
    
    class policy(LeggedRobotAMPCfgFlashSAC.policy):
        actor_hidden_dim = 128
        actor_num_blocks = 2
        critic_hidden_dim = 256
        critic_num_blocks = 2
        # Use paper defaults: 101 bins on [-5, 5] with reward normalization

    class algorithm(LeggedRobotAMPCfgFlashSAC.algorithm):
        amp_replay_buffer_size = 245_760  # 1024 envs × 24 steps × 10 rollouts
        disc_lr = 1e-4
        batch_size = 2048
        utd_ratio = 2  # Back to 2 for faster iterations
        # Paper values (Table 9)
        tau = 0.01
        init_alpha = 0.01
        temp_target_sigma = 0.15
        noise_zeta_mu = 2.0
        noise_zeta_max = 16
        normalize_rewards = True       # paper Eq 6
        n_step = 1
        # Cosine decay LR: 3e-4 → 1.5e-4 over full training (halved steps)
        learning_rate_end = 1.5e-4
        learning_rate_decay_steps = 10000

    class runner(LeggedRobotAMPCfgFlashSAC.runner):
        amp_reward_coef = 1.5
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 245_760  # 1024 envs × 24 steps × 10 rollouts
        amp_discr_hidden_dims = [1024, 512]
        amp_task_reward_lerp = 0.6  # Stronger AMP signal (was 0.75)
        # Dynamic warmup based on actual --num_envs at runtime
        warmup_steps_per_env = 5  # Will be: actual_num_envs * 5 in runner
        replay_buffer_size = 10_000_000

        max_iterations = 10030  # Halved from 20060
        save_interval = 200
        run_name = f"k1_amp_flashsac_{SIMULATOR}"
        experiment_name = "k1_amp_flashsac"


# Keep env reward overrides aligned with SAC AMP tuning.
class K1AMPFlashSACEnvCfg(K1AMPSACEnvCfg):
    class env(K1AMPSACEnvCfg.env):
        num_envs = 1024
