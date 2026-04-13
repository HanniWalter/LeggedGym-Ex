# This file contains template configuration classes for legged robot tasks
# These classes serve as base templates for task-specific configurations
# Author: Genesis LR

from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# ----- Template configuration for AMP tasks -----#
from legged_gym import LEGGED_GYM_ROOT_DIR
import glob

MOTION_FILES = glob.glob(LEGGED_GYM_ROOT_DIR + "/resources/reference_motion/*")

class LeggedRobotAMPCfg(LeggedRobotCfg):
    
    class env(LeggedRobotCfg.env):
        amp_motion_files = MOTION_FILES
    
    class init_state(LeggedRobotCfg.init_state):
        # whether to initialize the robot with the reference motion
        reference_state_initialization = True
        reference_state_initialization_prob = 0.7
        

class LeggedRobotAMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'AMPRunner'
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        amp_replay_buffer_size = 1000000
        disc_lr = 1e-4

    class runner( LeggedRobotCfgPPO.runner ):
        algorithm_class_name = 'PPO_AMP'
        
        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_discr_hidden_dims = [1024, 512]
        amp_task_reward_lerp = 0.3                 # Task reward blending ratio


# ----- Template configuration for Teacher-Student framework -----#
class LeggedRobotTSCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_observations = 48
        num_privileged_obs = None
        # for teacher-student framework
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = num_privileged_obs
        c_frame_stack = 5
        num_single_critic_obs = num_observations
        num_critic_obs = c_frame_stack * num_single_critic_obs

class LeggedRobotTSCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'TSRunner'
    class policy( LeggedRobotCfgPPO.policy ):
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_type = "MLP" # "MLP" or "TCN"
        history_encoder_hidden_dims = [256, 128]       # for MLP
        history_encoder_channel_dims = [1, 1, 1, 1]    # for TCN
        history_encoder_dilation = [1, 1, 2, 1]        # for TCN
        history_encoder_stride = [1, 2, 1, 2]          # for TCN
        history_encoder_final_layer_dim = 128          # for TCN
        kernel_size = 5
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # for encoder training
        encoder_lr = 1.e-3
        num_encoder_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticTS'
        algorithm_class_name = 'PPO_TS'


# ----- Template configuration for Teacher-Student with Depth -----#
class LeggedRobotTSDepthCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_camera_envs = 1 # number of envs with depth camera, starting from the first env
        num_observations = 45
        num_privileged_obs = None
        # for teacher-student framework
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
        num_latent_dims = num_privileged_obs
        c_frame_stack = 5
        num_single_critic_obs = num_observations
        num_critic_obs = c_frame_stack * num_single_critic_obs

    class normalization( LeggedRobotCfg.normalization):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            depth_image = 2.0
            height_measurements = 1.0
        clip_actions = 100.0
    
class LeggedRobotTSDepthCfgPPO(LeggedRobotCfgPPO):
    distillation = False # false -> teacher training, true -> student training
    runner_class_name = 'TSDepthRunner'
    class policy( LeggedRobotCfgPPO.policy ):
        critic_hidden_dims = [1024, 256, 128]
        actor_hidden_dims = [512, 256, 128]
        privilege_encoder_hidden_dims = [256, 128]
        cnn_input_channel = LeggedRobotTSDepthCfg.sensor.depth_camera_config.num_history
        cnn_channel_dims = [4, 8]
        cnn_strides = [1, 1]
        cnn_fc_layer_dims = [128]
        combination_mlp_dims = [128, 32]
        cnn_kernel_sizes = [5, 3]
        rnn_type = 'gru'
        rnn_hidden_size = 256
        rnn_num_layers = 1
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        encoder_lr = 2.e-4
        
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticTSDepth"
        algorithm_class_name = "PPO_TSDepth"
        teacher_model_path = "" # path to the teacher model checkpoint for distillation learning, necessary for student training


# ----- Template configuration for DreamWaQ -----#
class LeggedRobotDreamwaqCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_observations = 45  # num_obs
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = 16
        num_explicit_dims = 24  # base linear velocity
        num_decoder_output = num_observations
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 31 + 81 + 17 + 3
        num_privileged_obs = c_frame_stack * num_single_critic_obs

class LeggedRobotDreamwaqCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "DreamWaQRunner" # DreamWaQ Runner
    class policy( LeggedRobotCfgPPO.policy ):
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [256, 128]
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # for vae training
        encoder_lr = 2.e-4
        num_encoder_epochs = 1
        vae_kld_weight = 2.0

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQ"


# ----- Template configuration for CaT (Constraints as Termination) -----#
class LeggedRobotCTSCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_observations = 48
        num_privileged_obs = 94
        num_teacher = 1  # number of teacher envs
        # for teacher-student framework
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = num_privileged_obs
        c_frame_stack = 5
        num_single_critic_obs = num_observations
        num_critic_obs = c_frame_stack * num_single_critic_obs

class LeggedRobotCTSCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'CTSRunner'
    class policy( LeggedRobotCfgPPO.policy ):
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_hidden_dims = [256, 128]       # for MLP
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # for encoder training
        encoder_lr = 1.e-3
        num_encoder_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticCTS'
        algorithm_class_name = 'PPO_CTS'


# ----- Template configuration for Explicit Estimator -----#
class LeggedRobotEECfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        # Here the privileged_obs is actually critic_obs
        num_single_obs = 45
        frame_stack = 10    # number of frames to stack for obs_history
        num_estimator_features = int(num_single_obs * frame_stack)
        num_estimator_labels = 24
        c_frame_stack = 5
        num_single_critic_obs = num_single_obs + 31 + 81 + 17
        num_privileged_obs = c_frame_stack * num_single_critic_obs

class LeggedRobotEECfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'EERunner' # Explicit Estimator Runner
    class policy( LeggedRobotCfgPPO.policy ):
        estimator_hidden_dims = [256, 128]
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # for estimator training
        estimator_lr = 2.e-4
        num_estimator_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticEE'
        algorithm_class_name = 'PPO_EE'


# ----- Template configuration for SAC -----#
class LeggedRobotCfgSAC(LeggedRobotCfgPPO):
    """SAC configuration template for locomotion tasks."""
    seed = 1
    runner_class_name = 'SACRunner'
    class policy:
        clip_actions = 100.0
        init_noise_std = 1.0
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        activation = 'relu'
        use_layer_norm = True

    class algorithm:
        learning_rate = 3e-4
        alpha_lr = 3e-4
        gamma = 0.99
        tau = 0.005
        init_alpha = 1.0
        auto_alpha = True
        batch_size = 256
        utd_ratio = 1
        max_grad_norm = 1.0
        policy_delay = 1       # update actor every N critic updates
        use_adamw = False
        n_step = 1             # n-step returns (1 = standard TD)
        normalize_observations = True
        temporal_noise_steps = 0  # hold exploration noise for N steps (0 = disabled)

    class runner:
        policy_class_name = 'SACActorCritic'
        algorithm_class_name = 'SAC'
        num_steps_per_env = 24
        max_iterations = 3000
        replay_buffer_size = 1_000_000
        warmup_steps = 1000
        sync_wandb = False
        save_interval = 200
        experiment_name = 'test_sac'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None


# ----- Template configuration for SAC + AMP -----#
class LeggedRobotAMPCfgSAC(LeggedRobotCfgSAC):
    """SAC + AMP configuration template for motion imitation tasks."""
    runner_class_name = 'SACAMPRunner'

    class algorithm(LeggedRobotCfgSAC.algorithm):
        amp_replay_buffer_size = 1_000_000
        disc_lr = 1e-4

    class runner(LeggedRobotCfgSAC.runner):
        algorithm_class_name = 'SAC_AMP'

        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2_000_000
        amp_discr_hidden_dims = [1024, 512]
        amp_task_reward_lerp = 0.3


# ----- Template configuration for FastSAC (high UTD) -----#
class LeggedRobotCfgFastSAC(LeggedRobotCfgSAC):
    """FastSAC: SAC with high update-to-data (UTD) ratio for faster training.

    Key differences from standard SAC:
    - Larger batch size (1024 vs 256)
    - Higher UTD ratio (20 vs 1) – many gradient steps per env step
    - Larger replay buffer
    - Lower tau for slower target updates (compensates for high UTD)
    """
    class policy(LeggedRobotCfgSAC.policy):
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]

    class algorithm(LeggedRobotCfgSAC.algorithm):
        batch_size = 1024
        utd_ratio = 20
        tau = 0.005
        learning_rate = 3e-4
        alpha_lr = 1e-4

    class runner(LeggedRobotCfgSAC.runner):
        replay_buffer_size = 2_000_000
        warmup_steps = 5000
        num_steps_per_env = 1
        experiment_name = 'test_fastsac'


# ----- Template configuration for FastSAC + AMP -----#
class LeggedRobotAMPCfgFastSAC(LeggedRobotCfgFastSAC):
    """FastSAC + AMP for motion imitation with high UTD ratio."""
    runner_class_name = 'SACAMPRunner'

    class algorithm(LeggedRobotCfgFastSAC.algorithm):
        amp_replay_buffer_size = 2_000_000
        disc_lr = 1e-4

    class runner(LeggedRobotCfgFastSAC.runner):
        algorithm_class_name = 'SAC_AMP'

        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2_000_000
        amp_discr_hidden_dims = [1024, 512]
        amp_task_reward_lerp = 0.3
        experiment_name = 'test_fastsac_amp'


# ----- Template configuration for FlashSAC + AMP -----#
class LeggedRobotAMPCfgFlashSAC(LeggedRobotCfgSAC):
    """FlashSAC + AMP template config with residual/distributional SAC backend."""
    runner_class_name = 'FlashSACAMPRunner'

    class policy:
        clip_actions = 1.0
        init_noise_std = 1.0
        actor_num_blocks = 2
        actor_hidden_dim = 128
        critic_num_blocks = 2
        critic_hidden_dim = 256
        critic_num_bins = 101
        critic_min_v = -5.0
        critic_max_v = 5.0

    class algorithm:
        learning_rate = 3e-4
        alpha_lr = 3e-4
        gamma = 0.99
        tau = 0.01
        init_alpha = 0.01
        auto_alpha = True
        batch_size = 2048
        utd_ratio = 2
        max_grad_norm = 1.0
        policy_delay = 2
        use_adamw = False
        n_step = 1
        normalize_rewards = True
        normalize_observations = False
        temporal_noise_steps = 0
        noise_zeta_mu = 2.0
        noise_zeta_max = 16
        temp_target_sigma = 0.15
        critic_num_bins = 101
        critic_min_v = -5.0
        critic_max_v = 5.0
        learning_rate_end = 1.5e-4
        learning_rate_decay_steps = 0   # 0 = constant LR; set > 0 to enable cosine decay
        amp_replay_buffer_size = 1_000_000
        disc_lr = 1e-4

    class runner:
        policy_class_name = 'FlashSACActorCritic'
        algorithm_class_name = 'FlashSAC_AMP'
        num_steps_per_env = 24
        max_iterations = 3000
        replay_buffer_size = 1_000_000
        warmup_steps = 1000
        sync_wandb = False
        save_interval = 200
        experiment_name = 'test_flashsac_amp'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2_000_000
        amp_discr_hidden_dims = [1024, 512]
        amp_task_reward_lerp = 0.3
