import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

parser = argparse.ArgumentParser()
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #

# embedding parameters
parser.add_argument('--w2v_file', type=str, default='./inputs/w2v_200.txt', help='embedding file')
parser.add_argument('--all_data_file', type=str, default='./inputs/clause_keywords.csv', help='embedding file')
parser.add_argument('--train_file_path', type=str, default='./inputs/fold1_train.txt', help='train file path')
parser.add_argument('--dev_file_path', type=str, default='./inputs/fold1_test.txt', help='dev file path')
parser.add_argument('--test_file_path', type=str, default='./inputs/fold1_test.txt', help='test file path')
parser.add_argument('--embedding_dim', type=int, default=200, help='dimension of word embedding')
parser.add_argument('--embedding_dim_pos', type=int, default=50, help='dimension of position embedding')

# input struct
parser.add_argument('--max_clause_len', type=int, default=30, help='max number of tokens per sentence')
parser.add_argument('--max_doc_len', type=int, default=75, help='max number of tokens per documents')
# model struct
parser.add_argument('--lstm_hidden_dim', type=int, default=200, help='the number of features in the hidden state h')
parser.add_argument('--n_class', type=int, default=2, help='number of distinct class')

# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
parser.add_argument('--log_file_name', type=str, default='', help='name of log file')

# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
parser.add_argument('--num_epoch', type=int, default=30, help='number of train iter')
parser.add_argument('--scope', type=str, default='RNN', help='RNN scope')

# not easy to tune , a good posture of using data to train model is very important
parser.add_argument('--batch_size', type=int, default=32, help='number of example per batch')
parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
parser.add_argument('--embedding_drop', type=float, default=0.2, help='word embedding training dropout keep prob')
parser.add_argument('--softmax_drop', type=float, default=0, help='softmax layer dropout keep prob')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='l2 regularization')
parser.add_argument('--cause', type=float, default=1.000, help='lambda1')
parser.add_argument('--pos', type=float, default=1.00, help='lambda2')

# 设置一些我个人喜欢用的东西
parser.add_argument('--trainer_name', type=str, default='BaseTrainer', help='')
parser.add_argument('--model_name', type=str, default='IndependentBiLSTM', help='')
parser.add_argument('--do_train', type=bool, default=True, help='')
parser.add_argument('--do_test', type=bool, default=False, help='')
parser.add_argument('--device', type=str, default='cpu', help='')


args = parser.parse_args()
