'''
Author: Francois Milot & Jonathan Tremblay
Date: March 1, 2021.
'''

from math import copysign
import numpy as np
import itertools

class Board():
    # list of all 8 directions on the board, as (x,y) offsets
    list_direction_move = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
    list_direction_build = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
    list_direction_move_build = list(itertools.product(list_direction_move, list_direction_build))

    def __init__(self, n_board, n_tower):
        "Set up initial board configuration."
        # Initialize board dimension
        self.n_board = n_board
        self.n_tower = n_tower

        # Create the empty board array.
        self.pieces = np.zeros(((self.n_tower + 1), self.n_board, self.n_board)).astype(int)

        # Set up the initial 2 pieces.
        #TODO: ADD TWO OTHER PLAYERS
        self.pieces[-1][self.n_board-1][self.n_board-1] = 1
        self.pieces[-1][0][0] = -1

    def get_legal_moves_builds(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        """
        new_moves_builds = set()
        new_moves_builds_direction = set()

        for y in range(self.n_board):
            for x in range(self.n_board):
                if self.pieces[-1][x][y] == color:
                    new_move_build, new_move_build_direction = self.get_moves_for_square((x, y))
                    new_moves_builds.update(new_move_build)
                    new_moves_builds_direction.update(new_move_build_direction)
        return list(new_moves_builds), list(new_moves_builds_direction)

    def has_legal_moves_builds(self, color):
        for y in range(self.n_board):
            for x in range(self.n_board):
                if self.pieces[-1][x][y] == color:
                    new_move, _ = self.get_moves_for_square((x, y))
                    if len(new_move)>0:
                        return True
        return False

    def get_moves_for_square(self, square):
        """
        """
        # search all possible directions.
        x_orig, y_orig = square
        color = self.pieces[-1][x_orig][y_orig]

        moves_builds = []
        moves_builds_direction = []
        for direction_move in self.list_direction_move:
            move = self._discover_move(square, direction_move, color)

            if move is not None:
                for direction_build in self.list_direction_build:
                    build = self._discover_build(move, direction_build, color)
                    if build is not None:
                        move_build = (move, build)
                        moves_builds.append(move_build)
                        move_build_direction = (direction_move, direction_build)
                        moves_builds_direction.append(move_build_direction)
        return moves_builds, moves_builds_direction

    def _discover_move(self, origin, direction, color):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""

        x_orig, y_orig = origin
        x_dir, y_dir = direction
        x_sum, y_sum = x_orig + x_dir, y_orig + y_dir

        if self._is_legal_move(origin, direction, color):
            return (x_sum, y_sum)

        return None

    def _is_legal_move(self, origin, direction, color):
        x_orig, y_orig = origin
        x_dir, y_dir = direction
        x_sum, y_sum = x_orig + x_dir, y_orig + y_dir

        if not (x_sum >= self.n_board or y_sum >= self.n_board or x_sum < 0 or y_sum < 0): # boundaries of board
            if (self.pieces[-1][x_sum][y_sum] != -color): # no oppenent on the place where he is going
                if np.sum(self.pieces[:-1], axis=0)[x_sum][y_sum] != 4: # capped building
                    if np.sum(self.pieces[:-1], axis=0)[x_sum][y_sum] - np.sum(self.pieces[:-1], axis=0)[x_orig][y_orig] <= 1: # move only 1 building at a time
                        return True
        return False

    def _discover_build(self, origin, direction, color):
        x_orig, y_orig = origin
        x_dir, y_dir = direction
        x_sum, y_sum = x_orig + x_dir, y_orig + y_dir

        if self._is_legal_build(origin, direction, color):
            return (x_sum, y_sum)

        return None

    def _is_legal_build(self, origin, direction, color):
        x_orig, y_orig = origin
        x_dir, y_dir = direction
        x_sum, y_sum = x_orig + x_dir, y_orig + y_dir

        if (not (x_sum >= self.n_board or y_sum >= self.n_board or x_sum < 0 or y_sum < 0)): # boundaries of board
            if (self.pieces[-1][x_sum][y_sum] != -color): # no opponent on building
                if np.sum(self.pieces[:-1], axis=0)[x_sum][y_sum] != 4: # no capped building
                    return True
        return False

    def execute_move_build(self, move, build, color):
        # Move
        x_move, y_move = move
        self._remove_color(color)
        self.pieces[-1][x_move][y_move] = color

        # Build
        x_build, y_build = build
        level = np.sum(self.pieces[:-1], axis=0)[x_build][y_build].astype(int)
        self.pieces[level][x_build][y_build] = 1

    #TODO: FIX THIS
    def read_action(self, action, color):
        """
        Function to return the mapping of the action number to actual move and build action that is readable by us.
        :param n: The board dimension
        :param action: The raw action in numeric
        :return: move (format (x, y)), build (format (z, w))
        """

        for x in range(self.n_board):
            for y in range(self.n_board):
                if self.pieces[-1][x][y] == color:
                    x_orig, y_orig = x, y

        (x_move_dir, y_move_dir), (x_build_dir, y_build_dir) = self.list_direction_move_build[action]

        move = (x_orig + x_move_dir), (y_orig + y_move_dir)
        build = (x_orig + x_move_dir + x_build_dir), (y_orig + y_move_dir + y_build_dir)

        return move, build

    def _remove_color(self, color):
        for y in range(self.n_board):
            for x in range(self.n_board):
                if self.pieces[-1][x][y] == color:
                    self.pieces[-1][x][y] = 0

