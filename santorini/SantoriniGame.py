from __future__ import print_function
import sys

sys.path.append('..')
from Game import Game
from .SantoriniLogic import Board
import numpy as np


class SantoriniGame(Game):
    square_content = {
        -31: "3-X",
        -21: "2-X",
        -11: "1-X",
        -1: "0-X",
        +40: "CAP",
        +30: "3--",
        +20: "2--",
        +10: "1--",
        +0: "0--",
        +31: "3-O",
        +21: "2-O",
        +11: "1-O",
        +1: "0-O"
    }

    @staticmethod
    def getSquarePiece(piece):
        """
        Function to get the mapping from the board representation to the square content dictionary
        :param piece: Representation in number of the board square
        :return: The readable representation of the board square
        """
        return SantoriniGame.square_content[piece]

    def __init__(self, n):
        super().__init__()
        self.n = n

    def getInitBoard(self):
        """
        Function to get the initial board of the game
        :return: Initial board of the game
        """
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        """
        Function to get the dimension of the board.
        :return: (mapping of the square content x dim of board x dim of board)
        """
        return (13, self.n, self.n)

    def getActionSize(self):
        """
        Function to get the action space dimension.
        :return: 5x5 moves x 5x5 builds
        """
        return self.n ** 4

    def getNextState(self, board, player, action):
        """
        Function to get the next board from an action that was performed by the current player.
        :param board: The board of the game before the action to be performed
        :param player: The player (1, -1) that will perform the action
        :param action: The action to be perform (raw in our case)
        :return: (State after action, opponent player of current player (1,-1)
        """
        b = Board(self.n)
        b.pieces = np.copy(board)

        move, build = self.read_action(action)
        b.execute_move_build(move, build, player)

        return (b.pieces, -player)

    def read_action(self, action):
        """
        Function to return the mapping of the action number to actual move and build action that is readable by us.
        :param n: The board dimension
        :param action: The raw action in numeric
        :return: move (format (x, y)), build (format (z, w))
        """
        move = (int(action / self.n ** 3), int((action / self.n ** 2) % self.n))
        build = (int((action / self.n) % self.n), int(action % self.n))
        return move, build

    def getValidMoves(self, board, player):
        """
        Function that give back a one-hot encoded vector of all valid moves of the action space
        :param board: The state representation
        :param player: The player that will take the actions
        :return: One-hot encoded vector of all valid moves of the action space
        """
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves_builds(player)

        if len(legalMoves) == 0:
            return np.array(valids)

        for move, build in legalMoves:
            x_move, y_move = move
            x_build, y_build = build
            valids[(self.n ** 3) * x_move + (self.n ** 2) * y_move + (self.n) * x_build + y_build] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Function to get if the player is winning or losing (or none)
        :param board: The state representation
        :param player: The player that is looking if he won or not
        :return: 1 if the player has won, -1 if he lost, 0 if the game is still pending.
        """
        b = Board(self.n)
        b.pieces = np.copy(board)

        outcome_p1 = np.where(b.pieces == player * 31)
        if outcome_p1[0].size > 0:
            return 1
        outcome_p1 = np.where(b.pieces == -player * 31)
        if outcome_p1[0].size > 0:
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
        assert isinstance(board, np.ndarray)
        output = player * board
        for row in range(output.shape[0]):
            for col in range(output.shape[1]):
                if output[row][col] in [-10, -20, -30, -40]:
                    output[row][col] = output[row][col] * -1
        return output


    # TODO: Major work here
    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert (len(pi) == self.n ** 4)  # 1 for pass
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

    def stringRepresentationReadable(self, board):
        """
        Similar function as above
        :param board: The state representation
        :return: The state representation in string format
        """
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("    ", end="")
        for y in range(n):
            print(y, end="   ")
        print("")
        print("------------------------")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = board[y][x]  # get the piece to print
                if piece in [-10, -20, -30, -40]:
                    piece *= -1
                print(SantoriniGame.square_content[piece], end=" ")
            print("|")

        print("------------------------")

def board_checker(board):
    for i in range(board.shape[0]):
        for j in range(board.shape[0]):
            list_of_moves = list(SantoriniGame.square_content.keys())
            if board[i][j] not in list_of_moves:
                return False
    return True

def getNNForm(board):
    assert isinstance(board, np.ndarray), 'Only accepts numpy array representation'
    board_level_map = {key: idx for idx, key in enumerate(SantoriniGame.square_content.keys())}
    nn_board = np.zeros((len(SantoriniGame.square_content), board.shape[0], board.shape[1]))
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            square = board[row][col]
            if square in [-10, -20, -30, -40]:
                board_i = board_level_map[-square]
            else:
                board_i = board_level_map[square]
            nn_board[board_i, row, col] = 1.0
    return nn_board

