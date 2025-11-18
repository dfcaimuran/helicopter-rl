# Helicopter Reinforcement Learning

This project trains a PPO agent to pilot a retro helicopter game using Stable-Baselines3 and a custom Gymnasium environment.

## 🎮 Gameplay Preview
Below is a preview GIF generated from a recorded gameplay frame:

![Gameplay GIF](helicopter_run.gif)

## Getting Started

### Prerequisites
- Python 3.11+
- FFmpeg (optional, required for video export)

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
```bash
python train.py --n-envs 32 --total-timesteps 5000000 --save-freq 5000
```

## Evaluation
```bash
python eval.py --model tmp/rl_model_500000_steps.zip --out-video gameplay.mp4
```

## Play Manually
```bash
python helicopter_game.py
```

## Project Structure
- helicopter_game.py – Pygame implementation of the helicopter game  
- helicopter_env.py – Gymnasium environment wrapper  
- train.py – PPO training entry point  
- eval.py – Evaluation and video recording  
- assets/ – Sprites and fonts  
