import imageio
import numpy as np
from stable_baselines3 import PPO
from helicopter_env import HelicopterEnv


def pad_frame_to_16(frame):
    """
    Pad the frame so that width and height are divisible by 16.
    This avoids FFmpeg resizing warnings.
    """
    h, w, _ = frame.shape

    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    if pad_h == 0 and pad_w == 0:
        return frame  # Already OK

    # Pad on the right and bottom using black pixels
    padded = np.pad(
        frame,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=0
    )
    return padded


# def record_video(
#     model_path="tmp/rl_model_100100000_steps.zip", # best model  
#     # model_path="tmp/rl_model_1000000_steps.zip",
#     video_path="helicopter_run.mp4",
#     fps=60,
#     max_steps=30000,
# ):
#     print(f"Loading model: {model_path}")
#     model = PPO.load(model_path)

#     # Important: must use rgb_array mode for video recording
#     env = HelicopterEnv(render_mode="rgb_array")

#     obs, info = env.reset()
#     frames = []

#     print("Recording video...")

#     for step in range(max_steps):
#         # Choose action from trained policy
#         action, _ = model.predict(obs, deterministic=True)

#         # Step environment
#         obs, reward, terminated, truncated, info = env.step(action)

#         # Get visual frame as an RGB array
#         frame = env.render()

#         # Fix size so width/height divisible by 16 (remove FFMPEG warning)
#         frame = pad_frame_to_16(frame)

#         frames.append(frame)

#         if terminated or truncated:
#             print(f"Episode ended at step {step}")
#             break

#     env.close()

#     print(f"Saving video to: {video_path}")
#     imageio.mimsave(video_path, frames, fps=fps)
#     print("Done! No resizing warnings.")

# import imageio
# import numpy as np
# from stable_baselines3 import PPO
# from helicopter_env import HelicopterEnv


def record_video(
    model_path="tmp/rl_model_5000000_steps.zip",
    video_path="helicopter_run.mp4",
    gif_path="helicopter_run.gif",
    fps=60,
    max_steps=30000,
):
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    # RGB render mode
    env = HelicopterEnv(render_mode="rgb_array")

    obs, info = env.reset()
    frames = []

    print("Recording video...")

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()  # (H, W, 3)
        frames.append(frame)

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    env.close()

    print(f"Saving MP4 video to: {video_path}")
    imageio.mimsave(video_path, frames, fps=fps, macro_block_size=None)

    print(f"Saving GIF to: {gif_path}")
    imageio.mimsave(
        gif_path, 
        frames, 
        fps=30, 
        loop=0,
        palettesize=256,
        subrectangles=True,
    )

    print("Done!")


if __name__ == "__main__":
    record_video()