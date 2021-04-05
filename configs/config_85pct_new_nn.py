"""

"""
from utils import dotdict
import torch
import os

config_main = dotdict({
    'numIters': 25,
    'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,  #
    'updateThreshold': 0.6,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 20,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'nb_of_new_model_for_random_player': 1,
    'nb_of_game_agaisnt_random_player': 50,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'create_log_file': True,
    'log_file_location': f'/home/john/PycharmProjects/ift-6756-alphazero/logs/',
    'log_file_name': f'{os.path.basename(__file__)}_log.txt',
    'log_run_name': f'{os.path.basename(__file__)}'
})
config_nn = dotdict({
    'lr': 0.001,
    'dropout': 0.2,
    'epochs': 10,
    'batch_size': 1024,
    'cuda': torch.cuda.is_available(),
    'num_channels': 256,
})
