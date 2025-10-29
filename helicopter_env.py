from typing import Literal

import numpy as np
import pygame
from gymnasium import Env, spaces
from helicopter_game import HelicopterGame


class HelicopterEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    MAX_TUNNEL_STEPS = 4

    def __init__(self, render_mode: Literal["human", "rgb_array"] = "human"):
        super().__init__()
        self.render_mode = render_mode
        self.game = HelicopterGame(render_mode=render_mode)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2 + self.MAX_TUNNEL_STEPS * 2,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        observation = self.__get_obs()
        info = self.__get_info()
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action)
        self.game.action = int(action)
        self.game.step()

        observation = self.__get_obs()
        reward = 0.0 if self.game.game_over else 1.0
        terminated = self.game.game_over
        truncated = False
        info = self.__get_info()
        return observation.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.game.draw()
            return None
        if self.render_mode == "rgb_array":
            original_surface = self.game.surface
            original_screen = self.game.screen
            self.game.surface = pygame.Surface((self.game.WIDTH, self.game.HEIGHT))
            self.game.screen = None
            self.game.draw()
            array = pygame.surfarray.array3d(self.game.surface).transpose(1, 0, 2)
            self.game.surface = original_surface
            self.game.screen = original_screen
            return array

    def __get_info(self):
        return {"game_over": self.game.game_over}

    def __get_obs(self):
        player = np.array(
            [
                self.game.helicopter_pos_y / self.game.HEIGHT,
                self.game.helicopter_speed_y / self.game.HELICOPTER_SPEED_Y_MAX * 0.5
                + 0.5,
            ],
            dtype=np.float32,
        )

        tunnel = np.full((self.MAX_TUNNEL_STEPS, 2), [1.0, 0.5], dtype=np.float32)
        for index, t in enumerate(self.game.tunnel[: self.MAX_TUNNEL_STEPS]):
            tunnel[index] = (
                (t.x + self.game.WIDTH) / (self.game.WIDTH * 3),
                t.y / self.game.HEIGHT,
            )
        return np.concatenate([player, tunnel.ravel()])
