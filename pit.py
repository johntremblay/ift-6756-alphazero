import Arena
from MCTS import MCTS
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniPlayers import *
from santorini.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

g = SantoriniGame(5)

# all players
rp1 = RandomPlayer(g, 1).play
rp2 = RandomPlayer(g, -1).play
hp1 = HumanSantoriniPlayer(g, 1).play
hp2 = HumanSantoriniPlayer(g, -1).play

# # nnet players
# n1 = NNet(g)
# n1.load_checkpoint('./temp/','best.pth.tar')
# args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
# mcts1 = MCTS(g, n1, args1)
# n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
#
# if human_vs_cpu:
#     player2 = hp
# else:
#     n2 = NNet(g)
#     n2.load_checkpoint('./temp/', 'best.pth.tar')
#     args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
#     mcts2 = MCTS(g, n2, args2)
#     n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
#
#     player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(rp1, rp2, g, display=SantoriniGame.display)

print(arena.playGames(10, verbose=True))
