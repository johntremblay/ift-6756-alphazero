'''
Author: Francois Milot & Jonathan Tremblay
Date: March 1, 2021.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''

from math import copysign
import copy
import numpy as np


class Board():
    # list of all 8 directions on the board, as (x,y) offsets
    __directions_move = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
    __directions_build = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]


    def __init__(self, n):
        """Set up initial board configuration."""
        super().__init__()
        self.n = n
        # Create the empty board list
        self.pieces = [None] * self.n
        for i in range(self.n):
            self.pieces[i] = [0] * self.n

        # Set up the initial 2 pieces.
        # TODO: ADD TWO OTHER PLAYERS
        self.pieces[self.n - 1][self.n - 1] = 1
        self.pieces[0][0] = -1

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]


    def get_legal_moves_builds(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        """
        new_moves_builds = set()

        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] in [i * color for i in [1, 11, 21]]:
                    new_moves_builds.update(self.get_moves_for_square((x, y)))
        return list(new_moves_builds)


    def has_legal_moves_builds(self, color):
        """
        This returns if a given player has legal moves to do or not
        :param color: int representation of a player (1 or -1)
        :return: bool if a player has legal moves or not
        """

        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] in [i * color for i in [1, 11, 21]]:
                    newmoves = self.get_moves_for_square((x, y))
                    if len(newmoves) > 0:
                        return True
        return False


    def get_moves_for_square(self, square):
        """
        Given a location as (x, y) ie: (row, col), this method checks where it is possible to move and build
        returns: a list of tuple of all the possible move and build: [((move_x, move_y), (build_x, build_y))...]
        """
        # search all possible directions.
        moves_builds = []
        for direction_move in self.__directions_move:
            move = self._discover_move(square, direction_move)
            if move is not None:
                board = copy.deepcopy(self.pieces)
                board = self._execute_move_any_board(board=board, move=move, color=np.sign(board[square[0], square[0]]))
                for direction_build in self.__directions_build:
                    build = self._discover_build_any_board(board=board, origin=move, direction=direction_build)
                    if build is not None:
                        move_build = (move, build)
                        moves_builds.append(move_build)
        return moves_builds

    @staticmethod
    def _execute_move_any_board(board, move, color):
        """
        private method, need to be sure the move is legal otherwise crash
        :param board:
        :param move:
        :return:
        """
        x_move, y_move = move
        Board._remove_color_any(board, color)
        board[x_move][y_move] = color * (board[x_move][y_move] + 1)
        return board


    def execute_move_build(self, move, build, color):
        # Move
        x_move, y_move = move
        self._remove_color(color)
        self[x_move][y_move] = color * (self[x_move][y_move] + 1)

        # Build
        x_build, y_build = build
        self[x_build][y_build] += 10


    def _remove_color(self, color):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] in [i * color for i in [1, 11, 21, 31]]:
                    self[x][y] = (self[x][y] - color) * color


    @staticmethod
    def _remove_color_any(board, color):
        for y in range(board.shape[1]):
            for x in range(board.shape[0]):
                if board[x][y] in [i * color for i in [1, 11, 21, 31]]:
                    board[x][y] = (board[x][y] - color) * color
        return board


    def _discover_move(self, origin, direction):
        """
        Returns the endpoint for a move, starting at the given origin,
        moving by the given increment.

        :returns None is move is not legal. Returns the new position after the move if legal
        """

        x_orig, y_orig = origin
        x_dir, y_dir = direction

        color = int(copysign(1, self[x_orig][y_orig]))

        if self._is_legal_move(origin, direction, color):
            return (x_orig + x_dir, y_orig + y_dir)

        return None


    def _is_legal_move(self, origin, direction, color):
        """
        Takes as input an origin point (x, y), a direction and a color (player 1 or -1) and will
        mention if the move is legal or not
        :param origin: (x, y)
        :param direction: (+/- 1 or 0, +/-1 or 0)
        :param color: +1 or -1
        :return: bool
        """
        x_orig, y_orig = origin
        x_dir, y_dir = direction

        x_sum = x_orig + x_dir
        y_sum = y_orig + y_dir

        if not (x_sum >= self.n or y_sum >= self.n or x_sum < 0 or y_sum < 0):  # boundaries of board
            if (self[x_sum][y_sum] not in [i * (-color) for i in [1, 11, 21, 31]]) and (
                    self[x_sum][y_sum] not in [40, -40]):  # players present or capped building
                if self[x_sum][y_sum] - color * (self[x_orig][y_orig] - color) <= 10:  # can only climb 1 floor at a time so need a max dif of 10
                    return True
        return False


    def _discover_build(self, origin, direction, board):
        """
        Returns the endpoint for a move, starting at the given origin,
        moving by the given increment.

        :returns None is move is not legal. Returns the new position after the move if legal
        """

        x_orig, y_orig = origin
        x_dir, y_dir = direction

        x_sum = x_orig + x_dir
        y_sum = y_orig + y_dir

        color = int(copysign(1, board[x_orig][y_orig]))

        if self._is_legal_build(x_sum, y_sum, color):
            return (x_sum, y_sum)

        return None


    @staticmethod
    def _discover_build_any_board(board, origin, direction):
        """
        Returns the endpoint for a move, starting at the given origin,
        moving by the given increment.

        :returns None is move is not legal. Returns the new position after the move if legal
        """

        x_orig, y_orig = origin
        x_dir, y_dir = direction

        x_sum = x_orig + x_dir
        y_sum = y_orig + y_dir

        color = int(copysign(1, board[x_orig][y_orig]))

        if Board._is_legal_build_any_board(board, x_sum, y_sum, color):
            return (x_sum, y_sum)

        return None


    def _is_legal_build(self, x_sum, y_sum, color):
        """
        check if it if possible to build at a given place
        :param x_sum:
        :param y_sum:
        :param color:
        :return:
        """
        if (not (x_sum >= self.n or y_sum >= self.n or x_sum < 0 or y_sum < 0)):  # should be within the boundaries
            if (self[x_sum][y_sum] not in [i * (-color) for i in [1, 11, 21, 31]]) and (
                    self[x_sum][y_sum] not in [40, -40]):  # cant build if other player occupies the spot or if capped
                return True
        return False


    @staticmethod
    def _is_legal_build_any_board(board, x_sum, y_sum, color):
        """
        check if it if possible to build at a given place
        :param x_sum:
        :param y_sum:
        :param color:
        :return:
        """
        n = board.shape[0]
        if (not (x_sum >= n or y_sum >= n or x_sum < 0 or y_sum < 0)):  # should be within the boundaries
            if (board[x_sum][y_sum] not in [i * (-color) for i in [1, 11, 21, 31]]) and (
                    board[x_sum][y_sum] not in [40, -40]):  # cant build if other player occupies the spot or if capped
                return True
        return False
