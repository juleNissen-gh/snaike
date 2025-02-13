"""
This module implements the game environment for a Snake-like game.

It provides the core game logic, including the snake's movement,
food spawning, and collision detection. The Environment class
can be used to create and manage game instances, which can be
used for both human play and AI training.
"""

import pygame as pg
import numpy as np
from random import randint
import itertools
from typing import Union


class Environment:
    """
    Represents the game environment for a Snake-like game.

    This class manages the game state, including the snake's position,
    food placement, and game boundaries. It provides methods for
    moving the snake, detecting collisions, and updating the game state.
    """

    # <editor-fold desc="Definitions... ">
    bounds = None
    clock = None
    window = None
    pg_initialized = False
    high_score: int = 0

    # these values can be customized
    ZOOM: int = 40  # how big the elements are
    ROUTE_SIZE: int = 1 * ZOOM  # size of each route
    BLOCK_SIZE: float = ((90 / 100) * ROUTE_SIZE)  # percentage of width and height of snake segment
    BOARD_PADDING: float = (50 / 100) * ROUTE_SIZE  # the amount of padding between the walls and game area
    BOARD_SIZE: int = 20  # number of rows and columns
    TRUNCATED_MATRIX_SIZE = 10  # New attribute for the truncated matrix size

    BLACK = (0, 0, 0)
    SNAKE_COLOR = (0, 255, 0)
    FOOD_COLOR = (255, 0, 0)

    LIVING_PENALTY: float = -0.02
    FOOD_REWARD: float = 1
    DEATH_PENALTY: float = -3
    WRONG_TURN_PENALTY: float = DEATH_PENALTY
    BORDER_PENALTY: float = LIVING_PENALTY * 3
    LEN_INPUTS: int = 18

    arrow_keys = {
        pg.K_LEFT: 0,
        pg.K_UP: 1,
        pg.K_RIGHT: 2,
        pg.K_DOWN: 3}

    action_map = {
        0: np.array([0, -1]),  # left
        1: np.array([1, -1]),  # up
        2: np.array([0, 1]),  # right
        3: np.array([1, 1])}  # down

    # </editor-fold>
    @staticmethod
    def color_square(color: tuple[int, int, int], pos: np.ndarray) -> None:
        """
        Colors a specified square on the game board.

        Args:
            color (tuple[int, int, int]): RGB color tuple.
            pos (np.ndarray): Position of the square to color.
        """

        pg.draw.rect(Environment.window, color, pg.Rect(
            Environment.ROUTE_SIZE * pos[0] + Environment.BOARD_PADDING,
            Environment.ROUTE_SIZE * pos[1] + Environment.BOARD_PADDING,
            Environment.BLOCK_SIZE, Environment.BLOCK_SIZE))

    @classmethod
    def set_board_size(cls, *, zoom: int, board_size: int, truncated_matrix_size: int = 10) -> None:
        """
        Adjusts the board size and related parameters before instantiating a game.

        Args:
            zoom (int): Zoom level for the game board.
            board_size (int): Size of the game board (number of cells).
            truncated_matrix_size (int, optional): Size of the truncated matrix. Defaults to 10.

        Raises:
            UserWarning: If the game is already initialized.
        """

        if cls.pg_initialized:
            raise UserWarning('Game is already initialized, definitions will not have effect.')

        cls.ZOOM = zoom
        cls.BOARD_SIZE = board_size
        cls.TRUNCATED_MATRIX_SIZE = truncated_matrix_size
        cls.ROUTE_SIZE = int(cls.ZOOM * 10 / cls.BOARD_SIZE)
        cls.BLOCK_SIZE = ((90 / 100) * cls.ROUTE_SIZE)
        cls.BOARD_PADDING = (50 / 100) * cls.ROUTE_SIZE

    @classmethod
    def init_pg(cls) -> None:
        """
        Initializes the Pygame window for visual representation of the game.
        This method should only be called for verbose games.
        """

        cls.pg_initialized = True
        pg.init()
        cls.bounds = (cls.BOARD_SIZE * cls.ROUTE_SIZE + cls.BOARD_PADDING * 2,) * 2
        cls.window = pg.display.set_mode(cls.bounds)
        pg.display.set_caption("Snake")
        cls.clock = pg.time.Clock()

    def __init__(self, *, verbose: Union[bool, int], pos: tuple[int, int] = (0, 0), init_pg=True):
        """
        Initializes a new game environment.

        Args:
            verbose (Union[bool, int]): Determines the verbosity of the game output.
            pos (tuple[int, int], optional): Initial position of the snake. Defaults to (0, 0).
            init_pg (bool, optional): Whether to initialize Pygame. Defaults to True.
        """

        if init_pg and not self.pg_initialized and verbose:
            self.init_pg()
        self.verbose = verbose
        self.snake: np.ndarray = np.array([[0, 0], [1, 0]]) + pos
        self.prev_dir = np.array([0, 1])
        self.temp_snake = None
        self.grow = True
        self.playing = True
        self.temp_playing = None
        self.food: np.ndarray = self.spawn_food(True)
        if self.verbose:
            self.color_square(Environment.SNAKE_COLOR, self.snake[0])
            self.color_square(Environment.SNAKE_COLOR, self.snake[-1])
            pg.display.update()

    def step(self, action: int, cannon=True) -> tuple[float, np.ndarray, np.ndarray, bool]:
        """
        Advances the game state by one step based on the given action.

        Args:
            action (int): The action to take (0: left, 1: up, 2: right, 3: down).
            cannon (bool, optional): Whether to save the step. Defaults to True.

        Returns:
            tuple: Contains (reward, next_state, next_b_matrix, is_playing).

        Raises:
            Exception: If the game has already ended.
        """

        if not self.playing:
            raise Exception('Game has already ended')

        if isinstance(action, int):
            action = self.action_map[action]
        reward = self.LIVING_PENALTY

        # copy all modified variables in case move shouldn't be saved
        legal_turn = True
        prev_snake = self.snake.copy()
        prev_food = self.food
        prev_playing = self.playing
        prev_grow = self.grow

        # check that the move is valid (no moving backwards)
        new_pos = np.copy(self.snake[-1])
        if not (abs(np.sum(self.prev_dir) - np.sum(action)) == 2):
            new_pos[action[0]] += action[1]
        else:
            reward += self.WRONG_TURN_PENALTY
            legal_turn = False
            new_pos[self.prev_dir[0]] += self.prev_dir[1]
            action = self.prev_dir
        self.snake = np.append(self.snake, [new_pos], axis=0)

        if np.any(new_pos == 0) | np.any(new_pos == Environment.BOARD_SIZE - 1):  # near edges
            reward += Environment.BORDER_PENALTY

        # Food protocol
        if np.all(self.food == self.snake[-1]):
            self.grow = True
            reward += self.FOOD_REWARD * legal_turn
            self.spawn_food(cannon)

        # Check for death
        if (any(np.unique(self.snake[1 * (not self.grow):], axis=0, return_counts=True)[1] > 1)) or \
                np.any((self.snake[-1] < 0) | (self.snake[-1] >= Environment.BOARD_SIZE)):
            self.playing = False
            reward += self.DEATH_PENALTY if legal_turn else 0
            Environment.high_score = max(Environment.high_score, len(self.snake) - 1)

        # Update screen
        elif self.verbose and cannon:
            self.update_snake()
            pg.display.update()

        if not self.grow:
            self.snake = self.snake[1:]

        next_state, next_b_matrix = self.get_state()

        if not cannon:  # reset values to previously
            self.grow = prev_grow
            self.snake = prev_snake
            self.food = prev_food
            self.playing, prev_playing = prev_playing, self.playing
        else:
            self.prev_dir = action
            self.grow = False
            prev_playing = self.playing

        return reward, next_state, next_b_matrix, prev_playing and legal_turn

    def step_all(self, action: int) -> tuple[np.ndarray, ...]:
        """
        Simulates the results of all possible actions from the current state.

        Args:
            action (int): The action that will actually be taken.

        Returns:
            tuple: Contains (rewards, next_states, next_b_matrices, playing_status) for all actions.
        """

        rewards = np.zeros(4)
        playing = np.zeros(4)
        next_states = np.zeros((4, self.LEN_INPUTS))
        next_b_matricies = np.zeros((4, self.TRUNCATED_MATRIX_SIZE, self.TRUNCATED_MATRIX_SIZE))
        for direction in [i for i in [0, 1, 2, 3] if i != action] + [action]:  # do the cannon move last
            rewards[direction], next_states[direction], next_b_matricies[direction], playing[direction] = \
                self.step(direction, cannon=(direction == action))

        return rewards, next_states, next_b_matricies, playing

    def update_snake(self):
        """
        Updates the visual representation of the snake on the game board.
        This method respects the growth state of the snake.
        """

        if not self.grow:
            self.color_square(Environment.BLACK, self.snake[0])
        self.color_square(Environment.SNAKE_COLOR, self.snake[-1])

    def spawn_food(self, cannon) -> Union[np.ndarray, None]:
        """
        Spawns food at a random available position on the game board.

        Args:
            cannon (bool): Whether this is a real spawn (True) or a simulation (False).

        Returns:
            np.ndarray or None: The position of the spawned food, or None if no space is available.
        """

        squares = (Environment.BOARD_SIZE ** 2) - 1 - np.size(self.snake, axis=0)  # finds amount of eligible tiles
        if squares < 0:
            return None
        pos = randint(0, squares)

        index = 0
        # finds tile that corresponds to the chosen one
        for i in itertools.product(range(Environment.BOARD_SIZE), repeat=2):
            if not np.any(np.all(self.snake == i, axis=1)):
                if index == pos:
                    self.food: np.ndarray = np.array(i)
                    break
                index += 1

        if self.verbose and cannon:
            self.color_square(self.FOOD_COLOR, self.food)
        return self.food

    def snake_without_impossible_collisions(self) -> np.ndarray:
        """
        Returns a subset of the snake's body that could potentially cause collisions.

        This method optimizes collision detection by excluding parts of the snake
        that are too far away to collide with the head in the next moves.

        Returns:
            np.ndarray: Array of snake body positions that could cause collisions.
        """

        return self.snake[:-1][np.array([i >= np.sum(np.abs(self.snake[-1] - self.snake[i]))
                                         for i in range(len(self.snake) - 1)])]

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:

        """
        Retrieves the current state of the game.

        The state includes information about food position, snake length,
        obstacles, and borders relative to the snake's head in the following order:

        food is E, S, W, N
        dist food
        len snake
        obstacle dist E, S, W, N, SE, NE, SW, NW
        border is E, S, W, N

        Returns:
            tuple: Contains (state_vector, board_matrix).
                state_vector (np.ndarray): A 1D array representing various game state features.
                board_matrix (np.ndarray): A 2D matrix representing the game board state.

        """
        matrix_pad_width = self.TRUNCATED_MATRIX_SIZE // 2 + bool(self.TRUNCATED_MATRIX_SIZE % 2)
        if not self.playing:
            return np.zeros(self.LEN_INPUTS), np.zeros((self.TRUNCATED_MATRIX_SIZE,) * 2)

        state = np.array([])
        head: np.ndarray = self.snake[-1]
        state = np.append(state, self.food - head > 0)
        state = np.append(state, self.food - head < 0)
        state = np.append(state, np.sum(np.abs(head - self.food)))
        state = np.append(state, np.size(self.snake - 1, axis=0))

        trimmed_snake = self.snake_without_impossible_collisions()

        # find distances to obstacles in 8 directions
        # I'm not even going to try to explain how it works (cause i may or may not have forgotten it)
        size = Environment.BOARD_SIZE
        for dir_ in (
                (size - head[0], 1, 0),
                (size - head[1], 0, 1),
                (head[0], -1, 0),
                (head[1], 0, -1),

                (size - max(head), 1, 1),
                (np.min((size - head[0], head[1])), 1, -1),
                (np.min((size - head[1], head[0])), -1, 1),
                (min(head), -1, -1)
        ):

            for i in range(1, dir_[0] + 1):
                if np.any(np.all(trimmed_snake ==
                                 (head + ((i * dir_[1],) + (i * dir_[2],))), axis=1)):
                    dist = i - 1
                    break
            else:  # hit a wall, not snake
                dist = dir_[0]
            state = np.append(state, dist)

        state = np.append(state, np.append(head-1 <= 0, head >= self.BOARD_SIZE-2))

        matrix = np.zeros((self.BOARD_SIZE,) * 2)
        matrix[trimmed_snake[:, 0], trimmed_snake[:, 1]] = 1

        matrix = np.pad(matrix, pad_width=1, mode='constant', constant_values=1)
        matrix = np.pad(matrix, pad_width=matrix_pad_width - 1, mode='constant', constant_values=0)

        x, y = head[0], head[1]
        matrix = matrix[x:x + self.TRUNCATED_MATRIX_SIZE, y:y + self.TRUNCATED_MATRIX_SIZE]

        return state, matrix
