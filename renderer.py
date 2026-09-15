import numpy as np
import pygame
from numpy.typing import NDArray

from snake import Game, Point


class GameRenderer:
    def __init__(
        self,
        game: Game,
        cell_size: int = 30,
        fps: int = 10,
    ):
        self.game = game
        self.cell_size = cell_size
        self.fps = fps

        self.window_width = game.width * cell_size
        self.window_height = game.height * cell_size

        pygame.init()

        self.screen = pygame.display.set_mode(
            (
                self.window_width,
                self.window_height,
            )
        )

        pygame.display.set_caption("Snake AI")

        self.clock = pygame.time.Clock()

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        return True

    def render(self) -> None:
        self.screen.fill((20, 20, 20))

        self._draw_grid()
        self._draw_food()
        self._draw_snake()

        pygame.display.flip()

        self.clock.tick(self.fps)

    def close(self) -> None:
        pygame.quit()

    def _draw_grid(self) -> None:
        for x in range(self.game.width + 1):
            pygame.draw.line(
                self.screen,
                (40, 40, 40),
                (
                    x * self.cell_size,
                    0,
                ),
                (
                    x * self.cell_size,
                    self.window_height,
                ),
            )

        for y in range(self.game.height + 1):
            pygame.draw.line(
                self.screen,
                (40, 40, 40),
                (
                    0,
                    y * self.cell_size,
                ),
                (
                    self.window_width,
                    y * self.cell_size,
                ),
            )

    def _draw_food(self) -> None:
        if self.game.food is None:
            return

        self._draw_cell(
            self.game.food,
            color=(220, 60, 60),
        )

    def _draw_snake(self) -> None:
        for index, part in enumerate(self.game.snake.body):
            if index == 0:
                color = (60, 220, 100)
            else:
                color = (40, 160, 80)

            self._draw_cell(
                part,
                color=color,
            )

    def _draw_cell(
        self,
        point: Point,
        color: tuple[int, int, int],
    ) -> None:
        padding = 2

        rect = pygame.Rect(
            point.x * self.cell_size + padding,
            point.y * self.cell_size + padding,
            self.cell_size - padding * 2,
            self.cell_size - padding * 2,
        )

        pygame.draw.rect(
            self.screen,
            color,
            rect,
            border_radius=4,
        )

    def capture_frame(self) -> NDArray[np.uint8]:
        frame: NDArray[np.uint8] = pygame.surfarray.array3d(self.screen)

        return frame.swapaxes(0, 1).copy()
