from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from helicopter_env import HelicopterEnv
from stable_baselines3 import PPO

env = HelicopterEnv(render_mode=None)
env = Monitor(env)  # 👈 这一行

model = PPO.load("tmp/rl_model_100000000_steps.zip")  # 路径换成你的

mean_reward, std_reward = evaluate_policy(
    model, env,
    n_eval_episodes=50,
    deterministic=True
)

print(f"Mean reward over 50 episodes: {mean_reward:.2f} ± {std_reward:.2f}")