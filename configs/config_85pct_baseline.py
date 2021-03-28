"""

"""
from utils import dotdict
import torch

config_main = dotdict({
    'numIters': 25,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 5,        #
    'updateThreshold': 0.5,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 3,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/temp/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20
})

config_nn = dotdict({
        'lr': 0.001,
        'dropout': 0.1,
        'epochs': 25,
        'batch_size': 256,
        'cuda': torch.cuda.is_available(),
        'num_channels': 512,
    })
