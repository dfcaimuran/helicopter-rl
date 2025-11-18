import argparse
import json
import os
import time

import imageio.v2 as imageio
from helicopter_env import HelicopterEnv
# from utils.video import VideoWriter


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--out-images", type=str)
    parser.add_argument("--out-video", type=str)
    parser.add_argument("--generate-metadata", action="store_true")
    parser.add_argument("--action", type=int, choices=[0, 1])
    args = parser.parse_args()

    render_mode = "rgb_array" if args.out_dir else "human"
    env = HelicopterEnv(render_mode=render_mode)
    env.reset()
    n_steps = 1000
    metadata = []
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    # video_writer = (
        # VideoWriter(os.path.join(args.out_dir, args.out_video), fps=30)
        # if args.out_dir and args.out_video
        # else None
    # )
    for step in range(n_steps):
        print(f"Step {step + 1}")
        action = args.action if args.action is not None else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print("action=", action, "obs=", obs, "reward=", reward, "done=", done)
        if args.out_dir:
            frame = env.render()
            assert frame is not None
            image_file = args.out_images % step if "%" in args.out_images else None
            if args.out_images:
                assert image_file
                imageio.imwrite(
                    os.path.join(args.out_dir, image_file),
                    frame,
                    quality=100,
                )
            # if args.out_video:
            #     assert video_writer
            #     if terminated:
            #         for _ in range(15):
            #             video_writer.write(frame)
            #     else:
            #         video_writer.write(frame)
            if args.generate_metadata:
                metadata.append(
                    {
                        "frame": image_file,
                        "action": int(action),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
        else:
            env.render()
            time.sleep(0.02)
        if done:
            print("Goal reached!", "reward=", reward)
            break
    if args.out_dir and args.generate_metadata:
        with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    _main()
