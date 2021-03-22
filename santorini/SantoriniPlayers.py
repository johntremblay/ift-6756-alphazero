import numpy as np
from .SantoriniLogic import Board

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
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                move, build = Board.list_direction_move_build[i]
                print("[(", move[0], ", ", move[1], "), (", build[0], ", ", build[1],  end=")] ", sep='')
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 4:
                try:
                    x_move, y_move, x_build, y_build = [int(i) for i in input_a]
                    move_build_direction = ((x_move, y_move), (x_build, y_build))
                    a = Board.list_direction_move_build.index(move_build_direction)
                    if valid[a]:
                        break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a