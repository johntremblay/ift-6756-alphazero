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

# Instance of the game
g = SantoriniGame(5, 4)

# Players
rp = RandomPlayer(g).play
hp = HumanSantoriniPlayer(g).play

# # NNet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 15, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# Create an arena object to let agent play against each other
arena = Arena.Arena(n1p, rp, g, display=SantoriniGame.display)

print(arena.playGames(1000, verbose=False))
