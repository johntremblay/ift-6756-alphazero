import numpy as np


# SANTORINI: IN PROGRESS
class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a

class HumanSantoriniPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                move, build = self.game.read_action(i)
                print("[(", move[0], ", ", move[1], "), (", build[0], ", ", build[1],  end=")] ", sep='')
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 4:
                try:
                    x_move, y_move, x_build, y_build = [int(i) for i in input_a]
                    if ((0 <= x_move) and (x_move < self.game.n) and (0 <= y_move) and (y_move < self.game.n)) and \
                        ((0 <= x_build) and (x_build < self.game.n) and (0 <= y_build) and (y_build < self.game.n)):
                        a = (self.game.n ** 3) * x_move + (self.game.n ** 2) * y_move + (self.game.n * x_build) + y_build
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a
