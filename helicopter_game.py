import math
import random
import sys
from pathlib import Path
from typing import Literal
import os
import pygame


def _get_jagged_boundary(
    points,
    seed=0,
    y_offset=0,
    wave_phase=0,
    wave_amp=0,
):
    STEP_DIVISOR = 12
    OFFSET_BOUNDS = (-2, 2)

    jagged = []
    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        jagged.append(start)
        segment_length = end[0] - start[0]
        steps = max(1, int(segment_length) // STEP_DIVISOR)
        # Seed using tunnel segment's start position
        rng = random.Random(start[1] + seed)
        for step in range(1, steps):
            t = step / steps
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            jagged.append((x, y + rng.randint(*OFFSET_BOUNDS)))
    jagged.append(points[-1])

    return [
        (
            x,
            y + y_offset + math.sin((x + wave_phase) / 2.0) * wave_amp,
        )
        for x, y in jagged
    ]


class SpriteSheet:
    def __init__(self, path, frame_rects):
        self.sheet = pygame.image.load(str(path))
        self.rects = frame_rects
        self.frame_count = len(self.rects)

    def get_frame(self, index):
        index %= self.frame_count
        rect = self.rects[index]
        return self.sheet.subsurface(rect)


class HelicopterGame:
    WIDTH = 360  # Render target width in pixels
    HEIGHT = 240  # Render target height in pixels
    SCALE = 2  # Output scaling factor
    FPS = 60  # Framerate

    GRAVITY = 0.5  # Downward acceleration applied each frame
    THRUST = 0.3  # Upward acceleration when applying thrust

    HELICOPTER_WIDTH = 32  # Helicopter width in pixels
    HELICOPTER_HEIGHT = 16  # Helicopter height in pixels
    HELICOPTER_POS_X = WIDTH // 4  # Fixed horizontal position of the helicopter
    HELICOPTER_SPEED_X = 4  # Horizontal scrolling speed
    HELICOPTER_SPEED_Y_MAX = 10  # Maximum vertical speed

    TUNNEL_CENTER_OFFSET_MAX = 70  # Maximum vertical offset for tunnel center
    TUNNEL_SEGMENT_MIN = 80  # Minimum horizontal distance between tunnel points
    TUNNEL_SEGMENT_MAX = 120  # Maximum horizontal distance between tunnel points
    TUNNEL_HEIGHT = 100  # Vertical size of the tunnel corridor

    RESET_SPEED_ON_THRUST = True  # If the speed is downward, reset to 0 when thrusting

    def __init__(self, render_mode: Literal["human", "rgb_array"] = "human"):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        if not pygame.get_init():
            pygame.init()

        if render_mode == "human":
            self.screen = pygame.display.set_mode(
                (
                    self.WIDTH * self.SCALE,
                    self.HEIGHT * self.SCALE,
                )
            )
            pygame.display.set_caption("Helicopter Game")
            self.clock = pygame.time.Clock()

        self.surface = pygame.Surface((self.WIDTH, self.HEIGHT))

        asset_dir = Path(__file__).resolve().parent / "assets"
        self.helicopter_sprite = SpriteSheet(
            asset_dir / "helicopter.png",
            [
                pygame.Rect(0, 0, 29, 20),
                pygame.Rect(29, 0, 29, 20),
                pygame.Rect(0, 20, 29, 20),
                pygame.Rect(29, 20, 29, 20),
            ],
        )
        self.explosion_sprite = SpriteSheet(
            asset_dir / "explosion.png",
            [
                pygame.Rect(0, 0, 32, 24),
                pygame.Rect(32, 0, 32, 24),
                pygame.Rect(0, 0, 32, 24),
            ],
        )

        # font_path = str(asset_dir / "ark-pixel-12px-monospaced-zh_cn.ttf")
        # self.font = pygame.font.Font(font_path, 12 * 2)
        # self.info_font = pygame.font.Font(font_path, 12)
        # self.distance_font = pygame.font.Font(font_path, 18)

        font_path = asset_dir / "ark-pixel-12px-monospaced-zh_cn.ttf"
        if font_path.exists():
            font_path_str = str(font_path)
            self.font = pygame.font.Font(font_path_str, 12 * 2)
            self.info_font = pygame.font.Font(font_path_str, 12)
            self.distance_font = pygame.font.Font(font_path_str, 18)
        else:
            print(
                f"Warning: font file not found at {font_path}, "
                "using default system font instead."
            )
            self.font = pygame.font.SysFont("Arial", 12 * 2)
            self.info_font = pygame.font.SysFont("Arial", 12)
            self.distance_font = pygame.font.SysFont("Arial", 18)

        self.is_running = True
        self.show_debug_info = True

        self.reset()

    def draw(self):
        self.__draw_background()
        self.__draw_stars()
        self.__draw_tunnel()
        self.__draw_helicopter()
        self.__draw_explosion()
        self.__draw_distance_text()
        self.__draw_game_over()
        self.__draw_author()
        self.__draw_trail()
        self.__draw_speed_indicator()
        self.__draw_debug_info()

        if self.screen:
            scaled = pygame.transform.scale(
                self.surface,
                (
                    self.WIDTH * self.SCALE,
                    self.HEIGHT * self.SCALE,
                ),
            )
            self.screen.blit(scaled, (0, 0))
            pygame.display.flip()

    def reset(self):
        self.game_over = False
        self.action = 0  # 0: do nothing, 1: move up

        self.tunnel = [
            pygame.Vector2(0.0, self.HEIGHT / 2),
            pygame.Vector2(self.WIDTH // 2, self.HEIGHT / 2),
        ]
        self.__update_tunnel()

        self.helicopter_pos_y = self.HEIGHT / 2
        self.helicopter_speed_y = 0
        self.__trail = []
        self.distance = 0

        self.frame_index = 0
        self.explosion_sprite_index = 0

    def run(self):
        if self.render_mode == "human":
            while self.is_running:
                assert self.clock
                self.clock.tick(self.FPS)
                self.__handle_events()
                if not self.is_running:
                    break

                self.step()

                self.draw()
            pygame.quit()
            sys.exit()

    def step(self):
        if self.game_over:
            return

        self.frame_index += 1

        self.distance += self.HELICOPTER_SPEED_X

        self.__update_helicopter_pos()

        self.__update_tunnel()

        self.__check_collision()

        self.__update_trail()

    def __check_collision(self):
        center_y = None
        for i in range(len(self.tunnel) - 1):
            left = self.tunnel[i]
            right = self.tunnel[i + 1]
            if left.x <= self.HELICOPTER_POS_X <= right.x:
                ratio = (self.HELICOPTER_POS_X - left.x) / (right.x - left.x)
                center_y = left.y + (right.y - left.y) * ratio
                break
        assert center_y is not None, "Center y should be found"
        helicopter_top = self.helicopter_pos_y - self.HELICOPTER_WIDTH * 0.5
        helicopter_bottom = self.helicopter_pos_y + self.HELICOPTER_HEIGHT * 0.5

        if self.helicopter_pos_y < 0 or self.helicopter_pos_y > self.HEIGHT:
            self.game_over = True
            return
        tunnel_top = center_y - self.TUNNEL_HEIGHT * 0.5
        tunnel_bottom = center_y + self.TUNNEL_HEIGHT * 0.5
        if helicopter_top < tunnel_top or helicopter_bottom > tunnel_bottom:
            self.game_over = True

    def __draw_author(self):
        if self.show_debug_info:
            author_text = self.info_font.render("By Ross Ning", True, (255, 255, 255))
            self.surface.blit(
                author_text,
                (
                    self.WIDTH - author_text.get_width() - 5,
                    self.HEIGHT - author_text.get_height() - 5,
                ),
            )

    def __draw_background(self):
        self.surface.fill((58, 0, 109))

    def __draw_debug_info(self):
        if self.show_debug_info:
            info_pairs = [
                ("THROTTLE", "ON" if self.action == 1 else "OFF"),
                ("SPEED Y", f"{self.helicopter_speed_y:.0f}"),
                ("POS Y", f"{self.helicopter_pos_y:.0f}"),
            ]
            line_height = self.info_font.get_linesize()
            x = self.WIDTH - 5
            y = 5
            for label, value in info_pairs:
                line = f"{label:<5} : {value:>3}"
                text_surface = self.info_font.render(line, True, (255, 255, 255))
                rect = text_surface.get_rect(topright=(x, y))
                self.surface.blit(text_surface, rect)
                y += line_height

    def __draw_distance_text(self):
        distance_text = self.distance_font.render(
            f"飞行距离: {self.distance:,}", False, (255, 255, 255)
        )
        distance_text_rect = distance_text.get_rect()
        distance_text_rect.centerx = self.WIDTH // 2
        distance_text_rect.top = 10
        bulletin_rect = pygame.Rect(
            0, 0, self.WIDTH // 2, distance_text_rect.height + 6
        )
        bulletin_rect.center = distance_text_rect.center
        overlay = pygame.Surface(bulletin_rect.size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.surface.blit(overlay, overlay.get_rect(center=bulletin_rect.center))
        self.surface.blit(distance_text, distance_text_rect)

    def __draw_explosion(self):
        if (
            self.game_over
            and self.explosion_sprite_index // 8 < self.explosion_sprite.frame_count
        ):
            explosion_frame = self.explosion_sprite.get_frame(
                self.explosion_sprite_index // 8
            )
            rect = explosion_frame.get_rect()
            rect.center = (self.HELICOPTER_POS_X, int(self.helicopter_pos_y))
            self.surface.blit(explosion_frame, rect)
            self.explosion_sprite_index += 1

    def __draw_game_over(self):
        if self.game_over:
            text = self.font.render("挑战失败", False, (255, 0, 0))
            rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.surface.blit(text, rect)

    def __draw_helicopter(self):
        helicopter_frame = self.helicopter_sprite.get_frame(
            self.frame_index // 2 % 2
            if self.helicopter_speed_y < 0
            else (2 + self.frame_index % 2)
        )

        if helicopter_frame:
            rect = helicopter_frame.get_rect()
            rect.center = (self.HELICOPTER_POS_X, int(self.helicopter_pos_y) - 4)
            self.surface.blit(helicopter_frame, rect)

    def __draw_stars(self):
        star_spacing = 16
        column_offset = self.distance // 8 // star_spacing
        intra_offset = self.distance // 8 % star_spacing
        columns_needed = self.WIDTH // star_spacing + 3
        for column_index in range(columns_needed):
            world_column = column_offset + column_index
            rng = random.Random(world_column)
            star_count = 1 + rng.randint(0, 2)
            for _ in range(star_count):
                x = (
                    column_index * star_spacing
                    - intra_offset
                    + rng.randint(0, star_spacing - 1)
                )
                y = rng.randint(0, self.HEIGHT) - int(self.helicopter_pos_y) // 16
                if 0 <= x < self.WIDTH:
                    self.surface.set_at(
                        (int(x), int(y)),
                        (255, 0, 255) if star_count % 2 == 0 else (255, 187, 255),
                    )

    def __draw_tunnel(self):
        layer_count = 4
        layer_colors = [
            (148 - 15 * i, 115 - 13 * i, 24 - 3 * i) for i in range(layer_count)
        ]

        for sign, closing_points in (
            (-1, [(self.WIDTH, 0), (0, 0)]),
            (1, [(self.WIDTH, self.HEIGHT), (0, self.HEIGHT)]),
        ):
            boundary = [
                (
                    pt.x,
                    pt.y + sign * self.TUNNEL_HEIGHT * 0.5,
                )
                for pt in self.tunnel
            ]
            y_offsets = [sign * i * i * 4 for i in range(layer_count)]
            for i, (offset, color) in enumerate(zip(y_offsets, layer_colors)):
                boundary_points = _get_jagged_boundary(
                    boundary,
                    seed=i,
                    y_offset=offset,
                    wave_phase=self.distance + i * 8,
                    wave_amp=i * 8,
                )
                polygon = boundary_points + closing_points
                pygame.draw.polygon(self.surface, color, polygon)

    def __draw_trail(self):
        if len(self.__trail) > 1:
            pygame.draw.lines(self.surface, (255, 0, 0), False, self.__trail, 1)

    def __draw_speed_indicator(self):
        indicator_scale = 10
        x = self.HELICOPTER_POS_X
        y = self.helicopter_pos_y
        y1 = y + self.helicopter_speed_y * indicator_scale
        color = (0, 255, 0)
        pygame.draw.line(self.surface, color, (x, y), (x, y1), 1)
        if self.helicopter_speed_y:
            arrow_size = 4
            direction = 1 if self.helicopter_speed_y > 0 else -1
            pygame.draw.line(
                self.surface,
                color,
                (x, y1),
                (x - arrow_size, y1 - direction * arrow_size),
                1,
            )
            pygame.draw.line(
                self.surface,
                color,
                (x, y1),
                (x + arrow_size, y1 - direction * arrow_size),
                1,
            )

    def __handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.is_running = False
                if event.key in (pygame.K_r, pygame.K_SPACE) and self.game_over:
                    self.reset()

        keys = pygame.key.get_pressed()
        self.action = 1 if keys[pygame.K_SPACE] else 0

    def __update_helicopter_pos(self):
        if self.action == 1:
            if self.RESET_SPEED_ON_THRUST and self.helicopter_speed_y > 0:
                self.helicopter_speed_y = 0
            self.helicopter_speed_y -= self.THRUST
        else:
            self.helicopter_speed_y += self.GRAVITY
        self.helicopter_speed_y = max(
            -self.HELICOPTER_SPEED_Y_MAX,
            min(self.HELICOPTER_SPEED_Y_MAX, self.helicopter_speed_y),
        )

        self.helicopter_pos_y += self.helicopter_speed_y

    def __update_tunnel(self):
        for pt in self.tunnel:
            pt.x -= self.HELICOPTER_SPEED_X

        while self.tunnel[-1].x < self.WIDTH:
            self.tunnel.append(
                pygame.Vector2(
                    self.tunnel[-1].x
                    + random.randint(self.TUNNEL_SEGMENT_MIN, self.TUNNEL_SEGMENT_MAX),
                    self.HEIGHT * 0.5
                    + random.randint(
                        -self.TUNNEL_CENTER_OFFSET_MAX, self.TUNNEL_CENTER_OFFSET_MAX
                    ),
                )
            )
        while self.tunnel[1].x < 0:
            self.tunnel.pop(0)

    def __update_trail(self):
        for i, (x, y) in enumerate(self.__trail):
            self.__trail[i] = (x - self.HELICOPTER_SPEED_X, y)
        self.__trail.insert(0, (self.HELICOPTER_POS_X, self.helicopter_pos_y))
        self.__trail = [p for p in self.__trail if p[0] >= 0]


if __name__ == "__main__":
    game = HelicopterGame()
    game.run()
