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
        return SantoriniGame.square_content[piece]

    def __init__(self, n):
        super().__init__()
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # TODO: Possibly change repsentation to a lower dimension (5x5x5)
        return 13, self.n, self.n

    def getActionSize(self):
        # return number of actions
        # TODO: Possibly reduce to 8**2
        return self.n ** 4

    @staticmethod
    def getActionSize_any_board(n):
        # return number of actions
        return n ** 4

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board, player)
        # action must be a valid move
        b = Board(self.n)
        b.pieces = np.copy(board)

        move, build = self.read_action(action)
        b.execute_move_build(move, build, player)

        return (b.pieces, -player)

    @staticmethod
    def getNextState_any_board(board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        # TODO: if larger board
        b = Board(5)
        b.pieces = np.copy(board)

        move, build = SantoriniGame.read_action_any_board(action)
        b.execute_move_build(move, build, player)

        return (b.pieces, -player)

    @staticmethod
    def read_action_any_board(action):
        # TODO: Board size
        move = (int(action / 5 ** 3), int((action / 5 ** 2) % 5))
        build = (int((action / 5) % 5), int(action % 5))
        return move, build

    def read_action(self, action):
        move = (int(action / self.n ** 3), int((action / self.n ** 2) % self.n))
        build = (int((action / self.n) % self.n), int(action % self.n))
        return move, build

    def getValidMoves(self, board, player):
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

    @staticmethod
    def getValidMoves_any_board(board, player):
        # TODO Board size
        # return a fixed size binary vector
        valids = [0] * SantoriniGame.getActionSize_any_board(5)
        b = Board(5)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves_builds(player)

        if len(legalMoves) == 0:
            return np.array(valids)

        for move, build in legalMoves:
            x_move, y_move = move
            x_build, y_build = build
            valids[(5 ** 3) * x_move + (5 ** 2) * y_move + (5) * x_build + y_build] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        This method outputs if within a current state of the board if the game is finished and a player as won or
        not.
        :param board: np array representation of the board
        :param player: -1 or 1 to represent a player
        :return: int -1 or 1 or 0 depending if the game is finished or not
        """
        # TODO: Make sure that when we call this, is the game ended for a specific player
        b = Board(self.n)
        b.pieces = np.copy(board)

        outcome_p1 = np.where(b.pieces == player * 31)
        if outcome_p1[0].size > 0:
            return player
        outcome_p2 = np.where(b.pieces == -player * 31)
        if outcome_p2[0].size > 0:
            return -player

        if not b.has_legal_moves_builds(player):
            return -player
        elif not b.has_legal_moves_builds(-player):
            return player
        return 0

    def getCanonicalForm(self, board, player):
        # TODO: Really necessary?
        # return state if player==1, else return -state if player==-1
        assert isinstance(board, np.ndarray)
        output = board  # player * board
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
        return board.tostring()

    def stringRepresentationReadable(self, board):
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


# TODO: Move the below, since cannot be called and only a workaround of
# TODO: not a good canonicalboard representation
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
