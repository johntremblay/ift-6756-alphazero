from __future__ import print_function
import sys

sys.path.append('..')
from Game import Game
from .SantoriniLogic import Board
import numpy as np


class SantoriniGame(Game):

    def __init__(self, n_board, n_tower):
        super().__init__()
        self.n_board = n_board
        self.n_tower = n_tower

    def getInitBoard(self):
        """
        Function to get the initial board of the game
        :return: Initial board of the game
        """
        b = Board(self.n_board, self.n_tower)
        return np.array(b.pieces)

    def getBoardSize(self):
        """
        Function to get the dimension of the board.
        :return: (mapping of the square content x dim of board x dim of board)
        """
        return ((self.n_tower+1), self.n_board, self.n_board)

    def getActionSize(self):
        """
        Function to get the action space dimension.
        :return: moves (8) x builds (8)
        """
        return 8 * 8

    def getNextState(self, board, player, action):
        """
        Function to get the next board from an action that was performed by the current player.
        :param board: The board of the game before the action to be performed
        :param player: The player (1, -1) that will perform the action
        :param action: The action to be perform (raw in our case)
        :return: (State after action, opponent player of current player (1,-1)
        """
        b = Board(self.n_board, self.n_tower)
        b.pieces = np.copy(board)

        move, build = b.read_action(action, player) # possible problem here
        b.execute_move_build(move, build, player)

        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        """
        Function that give back a one-hot encoded vector of all valid moves of the action space
        :param board: The state representation
        :param player: The player that will take the actions
        :return: One-hot encoded vector of all valid moves of the action space
        """
        valids = [0]*self.getActionSize()

        b = Board(self.n_board, self.n_tower)
        b.pieces = np.copy(board)

        legalMoves, legalMoves_direction = b.get_legal_moves_builds(player)

        if len(legalMoves_direction) == 0:
            return np.array(valids)

        for move_build_direction in legalMoves_direction:
            valids[Board.list_direction_move_build.index(move_build_direction)] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Function to get if the player is winning or losing (or none)
        :param board: The state representation
        :param player: The player that is looking if he won or not
        :return: 1 if the player has won, -1 if he lost, 0 if the game is still pending.
        """
        b = Board(self.n_board, self.n_tower)
        b.pieces = np.copy(board)

        for x in range(self.n_board):
            for y in range(self.n_board):
                if b.pieces[-1][x][y] == player:
                    player_coordinate_x = x
                    player_coordinate_y = y
                if b.pieces[-1][x][y] == -player:
                    opponent_coordinate_x = x
                    opponent_coordinate_y = y

        if np.sum(b.pieces[:-1], axis=0)[player_coordinate_x][player_coordinate_y] == 3:
            return 1
        if np.sum(b.pieces[:-1], axis=0)[opponent_coordinate_x][opponent_coordinate_y] == 3:
            return -1

        if not b.has_legal_moves_builds(-player):
            return 1
        if not b.has_legal_moves_builds(player):
            return -1

        return 0

    def getCanonicalForm(self, board, player):
        """
        Function to switch the board from player's point of view. For instance, if player -1 is playing, he should see
        the board with his pawn as being flagged as 1 (not -1).
        :param board: The current state representation
        :param player: The player that is playing (1, -1)
        :return: The board switch from the player's point of view
        """
        b = Board(self.n_board, self.n_tower)
        b.pieces = np.copy(board)

        b.pieces[-1] = player * b.pieces[-1]
        return b.pieces

    # TODO: Major work here
    def getSymmetries(self, board, pi):
        # mirror, rotational
        return [(board, pi)]
        # pi_board = np.reshape(pi[:-1], (self.n, self.n))
        # l = []
        #
        # for i in range(1, 5):
        #     for j in [True, False]:
        #         newB = np.rot90(board, i)
        #         newPi = np.rot90(pi_board, i)
        #         if j:
        #             newB = np.fliplr(newB)
        #             newPi = np.fliplr(newPi)
        #         l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        # return l

    def stringRepresentation(self, board):
        """
        Convert board to string representation for MCTS
        :param board: The state representation
        :return: The state representation in string format
        """
        return board.tostring()

    @staticmethod
    def display(board):
        """
        Method to generate the readable board of the game
        :param board: the non-readable board of the game
        :return: readable board of the game
        """
        n_board = board.shape[1]
        print("    ", end="")
        for y in range(n_board):
            print(y, end="    ")
        print("")
        print("-----------------------------")
        for x in range(n_board):
            print(x, "|", end="")  # print the row #
            for y in range(n_board):
                piece = board[-1][x][y]  # get the piece to print
                floor = np.sum(board[:-1], axis=0)[x][y] if np.sum(board[:-1], axis=0)[x][y] > 0 else " "
                if piece == 0:
                    print(floor, "-", end="  ")
                elif piece == -1:
                    print(floor, "O", end="  ")
                else:
                    print(floor, "X", end="  ")
            print("|")

        print("-----------------------------")