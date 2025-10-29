import argparse
import subprocess
import time
from pathlib import Path

from helicopter_env import HelicopterEnv
from stable_baselines3 import PPO


class VideoWriter:
    def __init__(self, out_path: str | None = None, fps=60):
        self.fps = fps
        self.out_path = out_path
        self.video_proc = None

    def _start(self, frame):
        if self.out_path and frame is not None and self.video_proc is None:
            height, width = frame.shape[:2]
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{self.fps}",
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-crf",
                "1",
                "-pix_fmt",
                "yuv420p",
                self.out_path,
            ]
            self.video_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        if self.out_path and frame is not None:
            self._start(frame)
            if self.video_proc and self.video_proc.stdin:
                self.video_proc.stdin.write(frame.tobytes())

    def close(self):
        if self.video_proc and self.video_proc.stdin:
            self.video_proc.stdin.close()
        if self.video_proc:
            self.video_proc.wait()
            self.video_proc = None


def eval_agent(out_video=None, model_path=None):
    n_steps = 3000
    env = HelicopterEnv(render_mode="rgb_array" if out_video else "human")
    model = PPO.load(model_path, env=env)
    reset_result = env.reset()
    obs, _ = reset_result

    video_writer = VideoWriter(out_video) if out_video else None

    for step in range(n_steps):
        action, _ = model.predict(obs)
        print(f"Step {step + 1}")
        print("Action: ", action)
        step_result = env.step(action)
        obs, reward, terminated, truncated, info = step_result
        truncated = False
        print("obs=", obs, "reward=", reward, "done=", terminated or truncated)
        if video_writer:
            frame = env.render()
            if terminated:
                for _ in range(10):
                    env.step(action)
                    frame = env.render()
                    video_writer.write(frame)
            else:
                video_writer.write(frame)

        else:
            env.render()
        time.sleep(0.01)
        if terminated or truncated:
            print("Goal reached!", "reward=", reward)
            break
    if video_writer:
        video_writer.close()

    env.close()


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-video", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    model = args.model
    if model is None:
        log_dir = Path("tmp")
        model_files = sorted(log_dir.glob("*.zip")) if log_dir.exists() else []
        if not model_files:
            raise FileNotFoundError("No model checkpoint found; specify --model.")
        model = str(model_files[-1])

    eval_agent(out_video=args.out_video, model_path=model)


if __name__ == "__main__":
    _main()
