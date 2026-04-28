import os


from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.run_archiver import archive_run

def train(args):
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning')
    # Make environment and algorithm runner
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # Archive source code snapshot + resolved config JSON to log_dir
    log_dir = ppo_runner.log_dir
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        archive_run(log_dir, env_cfg, train_cfg, task_name=args.task)

    # Start training session
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        args.num_envs = 1
    train(args)
