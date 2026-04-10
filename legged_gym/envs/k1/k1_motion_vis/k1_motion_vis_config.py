from legged_gym import *
from legged_gym.envs.base.common_cfgs import K1FlatCommonCfg

"""
Booster K1 Motion Visualization environment configuration file.
"""
class K1MotionVisCfg(K1FlatCommonCfg):
    class env(K1FlatCommonCfg.env):
        num_observations = 75
        num_privileged_obs = num_observations + 3
        num_actions = 22
        episode_length_s = 10
        debug_draw_key_body_points = True # draw key body points for mimic tasks


class K1MotionVis20DofCfg(K1FlatCommonCfg):
    class env(K1FlatCommonCfg.env):
        num_observations = 69
        num_privileged_obs = num_observations + 3
        num_actions = 20
        episode_length_s = 10
        debug_draw_key_body_points = True # draw key body points for mimic tasks

    class init_state(K1FlatCommonCfg.init_state):
        default_joint_angles = {
            "ALeft_Shoulder_Pitch": 0.0,
            "Left_Shoulder_Roll": -1.3,
            "Left_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "ARight_Shoulder_Pitch": 0.0,
            "Right_Shoulder_Roll": 1.3,
            "Right_Elbow_Pitch": 0.0,
            "Right_Elbow_Yaw": 0.5,
            "Left_Hip_Pitch": -0.15,
            "Left_Hip_Roll": 0.0,
            "Left_Hip_Yaw": 0.0,
            "Left_Knee_Pitch": 0.3,
            "Left_Ankle_Pitch": -0.15,
            "Left_Ankle_Roll": 0.0,
            "Right_Hip_Pitch": -0.15,
            "Right_Hip_Roll": 0.0,
            "Right_Hip_Yaw": 0.0,
            "Right_Knee_Pitch": 0.3,
            "Right_Ankle_Pitch": -0.15,
            "Right_Ankle_Roll": 0.0,
        }

    class asset(K1FlatCommonCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/booster_robotics/K1/K1_20dof_slim_artificial.urdf"
        dof_names = [
            "ALeft_Shoulder_Pitch",
            "Left_Shoulder_Roll",
            "Left_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "ARight_Shoulder_Pitch",
            "Right_Shoulder_Roll",
            "Right_Elbow_Pitch",
            "Right_Elbow_Yaw",
            "Left_Hip_Pitch",
            "Left_Hip_Roll",
            "Left_Hip_Yaw",
            "Left_Knee_Pitch",
            "Left_Ankle_Pitch",
            "Left_Ankle_Roll",
            "Right_Hip_Pitch",
            "Right_Hip_Roll",
            "Right_Hip_Yaw",
            "Right_Knee_Pitch",
            "Right_Ankle_Pitch",
            "Right_Ankle_Roll",
        ]
        dof_armature = [
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.0478125,
            0.0339552,
            0.0282528,
            0.095625,
            0.0282528 * 2,
            0.0282528 * 2,
            0.0478125,
            0.0339552,
            0.0282528,
            0.095625,
            0.0282528 * 2,
            0.0282528 * 2,
        ]
