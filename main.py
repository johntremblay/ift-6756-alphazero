import logging
import coloredlogs
from Coach import Coach
from santorini.SantoriniGame import SantoriniGame as Game
from santorini.pytorch.NNet import NNetWrapper as nn

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

"""
Configuration minimal for now below
"""
from configs.config_85pct_baseline import config_main as args
from configs.config_85pct_baseline import config_nn as nn_args

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(5, 4)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g, args=nn_args)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args, nn_args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
