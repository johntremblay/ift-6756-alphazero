import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from utils import dotdict
import pandas as pd
from santorini.SantoriniPlayers import *
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import NNetWrapper as NNet
import time

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args, nn_args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, args=nn_args)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.log_file = os.path.join(args.log_file_location, args.log_file_name)
        self.nb_model_improv = 0
        self.nn_args = nn_args
        iters = self.args.numIters + 1
        rows_to_add = [0] * iters
        self.df_stats = pd.DataFrame({
            'iteration': [j for j in range(0, iters)],
            'nb_training_examples': rows_to_add,
            'avg_self_play_time_pr_game': rows_to_add,
            'arena_games': rows_to_add,
            'pct_games_won': rows_to_add,
            'new_nn_iteration_nb': rows_to_add,
            'winning_rate_vs_random': rows_to_add})

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            time_begin_iter = time.time()
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            self.df_stats = self.log_to_file(
                file=self.log_file,
                args=self.args,
                it=i,
                trainExamples=trainExamples,
                time_begin_iter=time_begin_iter,
                nwins=nwins,
                df_stats=self.df_stats,
                nb_model_improv=self.nb_model_improv)

            self.df_stats.to_feather(os.path.join(self.args.log_file_location, f"{self.args.log_run_name}.feather"))
            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nb_model_improv += 1
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                if self.nb_model_improv % self.args.nb_of_new_model_for_random_player == 0:
                    game_simul = SantoriniGame(5, 4)
                    rp = RandomPlayer(game_simul).play
                    n_simul = NNet(game_simul, self.nn_args)
                    n_simul.load_checkpoint('./temp/', 'best.pth.tar')
                    mcts_simul = MCTS(game_simul, n_simul, self.args)
                    n1_simul = lambda x: np.argmax(mcts_simul.getActionProb(x, temp=0))
                    arena_simul = Arena(n1_simul, rp, game_simul, display=False)
                    nnwins, _, _ = arena_simul.playGames(self.args.nb_of_game_agaisnt_random_player, verbose=False)
                    self.df_stats = self.log_to_file(
                        file=self.log_file,
                        args=self.args,
                        it=i,
                        trainExamples=trainExamples,
                        time_begin_iter=time_begin_iter,
                        nwins=nwins,
                        df_stats=self.df_stats,
                        nb_model_improv=self.nb_model_improv,
                        nb_game_rdm=self.args.nb_of_game_agaisnt_random_player,
                        nnwins=nnwins,
                        only_random=True)


    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    @staticmethod
    def log_to_file(file, args, it, trainExamples, time_begin_iter, nwins, df_stats, nb_model_improv, nb_game_rdm=100, nnwins=0, only_random=False):
        if not only_random:
            with open(file, 'a') as fp:
                fp.write(f"\n ### Iteration: {it}: \n")
                fp.write(
                    f"Number of self-play games: {args.numEps}\nNumber of training examples: {len(trainExamples)}\nAvg seconds by game:{round((time.time() - time_begin_iter) / args.numEps, 0)}\n")
                fp.write(
                    f"Arena games: {args.arenaCompare} \nPct of game won for new NN: {round(nwins / args.arenaCompare, 2)}\n")
                fp.close()
            df_stats.iloc[it, 1] = len(trainExamples)
            df_stats.iloc[it, 2] = round((time.time() - time_begin_iter) / args.numEps, 0)
            df_stats.iloc[it, 3] = args.arenaCompare
            df_stats.iloc[it, 4] = round(nwins / args.arenaCompare, 2)
            return df_stats
        else:
            with open(file, 'a') as fp:
                fp.write(
                    f"## Testing new NN vs random player (100 games):\nNew NN iteration number: {nb_model_improv}\nWinning rate versus random: {round(nnwins / 100, 2)}\n")
                fp.close()
                df_stats.iloc[it, 5] = nb_model_improv
                df_stats.iloc[it, 6] = round(nnwins / nb_game_rdm, 2)
                return df_stats
