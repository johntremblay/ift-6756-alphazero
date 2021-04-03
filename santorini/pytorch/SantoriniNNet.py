import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class SantoriniNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y, self.board_z = game.getBoardNNSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(SantoriniNNet, self).__init__()
        self.conv1 = nn.Conv3d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(args.num_channels, args.num_channels, 3, stride=1, padding=1)  # TODO: Maybe its here we want to fork for 2 actions

        # Here we will fork A
        self.conv5a = nn.Conv3d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv6a = nn.Conv3d(args.num_channels, args.num_channels, 3, stride=1)

        # Here we will fork B
        self.conv5b = nn.Conv3d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv6b = nn.Conv3d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm3d(args.num_channels)
        self.bn2 = nn.BatchNorm3d(args.num_channels)
        self.bn3 = nn.BatchNorm3d(args.num_channels)
        self.bn4 = nn.BatchNorm3d(args.num_channels)
        self.bn5 = nn.BatchNorm3d(args.num_channels)
        self.bn6 = nn.BatchNorm3d(args.num_channels)

        self.fc1a = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4)*(self.board_z-4), 13*512)
        self.fc1b = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4)*(self.board_z-4), 13*512)
        self.fc_bn1 = nn.BatchNorm1d(13*512)

        self.fc2a = nn.Linear(13 * 512, 256)
        self.fc2b = nn.Linear(13 * 512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)

        self.fc3a = nn.Linear(256, 8)
        self.fc3b = nn.Linear(256, 8)
        self.fc_bn3 = nn.BatchNorm1d(8)

        self.fc_comb = nn.Linear(64, 64)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y, self.board_z)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s_common = F.relu(self.bn4(self.conv4(s)))

        # Forking
        s_a = F.relu(self.bn5(self.conv5a(s_common)))
        s_a = F.relu(self.bn6(self.conv6a(s_a)))
        s_a = s_a.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4) * (self.board_z - 4))

        # Forking
        s_b = F.relu(self.bn5(self.conv5b(s_common)))
        s_b = F.relu(self.bn6(self.conv6b(s_b)))
        s_b = s_b.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4)*(self.board_z-4))

        s_a = F.dropout(F.relu(self.fc_bn1(self.fc1a(s_a))), p=self.args.dropout, training=self.training)
        s_b = F.dropout(F.relu(self.fc_bn1(self.fc1b(s_b))), p=self.args.dropout, training=self.training)

        s_a = F.dropout(F.relu(self.fc_bn2(self.fc2a(s_a))), p=self.args.dropout, training=self.training)
        s_b = F.dropout(F.relu(self.fc_bn2(self.fc2b(s_b))), p=self.args.dropout, training=self.training)

        conc = torch.cat((s_a, s_b), 1)
        v = self.fc4(conc)

        s_a = F.dropout(F.relu(self.fc_bn3(self.fc3a(s_a))), p=self.args.dropout, training=self.training)
        s_a = F.softmax(s_a)
        s_b = F.dropout(F.relu(self.fc_bn3(self.fc3b(s_b))), p=self.args.dropout, training=self.training)
        s_b = F.softmax(s_b)

        s_a = torch.transpose(s_a, 0, 1)
        s_common = torch.matmul(s_a, s_b)
        s_common = s_common.view(1, 64)
        # s_common = torch.cat((s_a, s_b), 1)
        pi = self.fc_comb(s_common)

        return F.softmax(pi, dim=1), torch.tanh(v)