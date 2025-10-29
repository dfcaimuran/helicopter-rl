import argparse
import os

from helicopter_env import HelicopterEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-envs",
        type=int,
        default=100,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000_000,
        help="Number of timesteps to train for",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=1000,
        help="Frequency (in steps) to save checkpoints",
    )
    args = parser.parse_args()

    vec_env = make_vec_env(
        HelicopterEnv,
        n_envs=args.n_envs,
        env_kwargs={"render_mode": "rgb_array"},
    )
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    vec_env = VecMonitor(vec_env, log_dir)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        device="cpu",
        batch_size=256,
    )
    tb_log_name = "ppo"
    if args.n_envs > 0:
        tb_log_name += f"_nenv{args.n_envs}"

    # Save a checkpoint periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path="./tmp/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=tb_log_name,
    )


if __name__ == "__main__":
    _main()
