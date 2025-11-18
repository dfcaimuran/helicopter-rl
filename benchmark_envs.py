import argparse
import time

from helicopter_env import HelicopterEnv
from stable_baselines3.common.env_util import make_vec_env


def benchmark_once(n_envs: int, steps_per_env: int, render_mode: str | None = None):
    """
    对给定 n_envs 跑 steps_per_env 步，统计 fps。
    总步数 = n_envs * steps_per_env
    """
    print(f"\n=== Benchmark: n_envs = {n_envs}, steps_per_env = {steps_per_env} ===")

    # 不需要渲染，纯算速度，所以 render_mode=None
    env_kwargs = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    vec_env = make_vec_env(
        HelicopterEnv,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
    )

    obs = vec_env.reset()
    total_steps = n_envs * steps_per_env

    start_time = time.time()
    for _ in range(steps_per_env):
        # 对每个 env 随机采样一个动作
        actions = [vec_env.action_space.sample() for _ in range(n_envs)]
        obs, rewards, dones, infos = vec_env.step(actions)
    elapsed = time.time() - start_time

    fps = total_steps / elapsed
    print(f"Total steps: {total_steps}")
    print(f"Elapsed time: {elapsed:.3f} s")
    print(f"→ FPS (environment steps / second): {fps:.1f}")

    vec_env.close()
    return fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-envs-list",
        type=str,
        default="1,4,8,16,32,64,100",
        help="Comma-separated list of n_envs to benchmark, e.g. '1,4,8,16,32,64,100'",
    )
    parser.add_argument(
        "--steps-per-env",
        type=int,
        default=2000,
        help="Number of steps to run per env for each benchmark point",
    )
    args = parser.parse_args()

    n_envs_list = [int(x) for x in args.n_envs_list.split(",")]

    print("Benchmarking env rollouts with random policy (no training)...")
    print(f"n_envs_list = {n_envs_list}")
    print(f"steps_per_env = {args.steps_per_env}")

    results = {}
    for n_envs in n_envs_list:
        fps = benchmark_once(n_envs, args.steps_per_env, render_mode=None)
        results[n_envs] = fps

    print("\n=== Summary ===")
    for n_envs in n_envs_list:
        print(f"n_envs = {n_envs:4d} -> {results[n_envs]:8.1f} fps")


if __name__ == "__main__":
    main()