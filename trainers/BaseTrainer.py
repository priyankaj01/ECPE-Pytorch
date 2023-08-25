"""
Base trainer
"""
import models
import torch.optim
import torch.nn as nn
import logging
import numpy as np
import time

from data.prepare_data import load_word2vector, load_data
from data.get_batch_data import get_batch_data
from utils.uitls import convert_one_hot_to_index
from metrics import compute_acc_precision_recall_f1

logger = logging.getLogger(__name__)


class BaseTrainer(object):
    def __init__(self, args):
        super(BaseTrainer, self).__init__()
        self.args = args
        self.idx2word, self.word2idx, self.word_embedding_lookup_table, self.position_embedding_lookup_table = load_word2vector(
            self.args.embedding_dim,
            self.args.embedding_dim_pos,
            self.args.all_data_file,
            self.args.w2v_file
        )

        tr_doc_id, tr_y_emotion, tr_y_cause, tr_y_pairs, tr_x, tr_clause_len, tr_doc_len = load_data(
            self.args.train_file_path, self.word2idx,
            self.args.max_doc_len, self.args.max_clause_len)
        # 正常用户不需要看这个 Develop dataset
        dev_doc_id, dev_y_emotion, dev_y_cause, dev_y_pairs, dev_x, dev_clause_len, dev_doc_len = load_data(
            self.args.dev_file_path, self.word2idx,
            self.args.max_doc_len, self.args.max_clause_len)
        te_doc_id, te_y_emotion, te_y_cause, te_y_pairs, te_x, te_clause_len, te_doc_len = load_data(
            self.args.test_file_path, self.word2idx,
            self.args.max_doc_len, self.args.max_clause_len)

        self.train_dataset = {
            'doc_id': tr_doc_id,
            'y_emotion': tr_y_emotion,
            'y_cause': tr_y_cause,
            'y_pairs': tr_y_pairs,
            'x': tr_x,
            'clause_len': tr_clause_len,
            'doc_len': tr_doc_len
        }
        self.dev_dataset = {
            'doc_id': dev_doc_id,
            'y_emotion': dev_y_emotion,
            'y_cause': dev_y_cause,
            'y_pairs': dev_y_pairs,
            'x': dev_x,
            'clause_len': dev_clause_len,
            'doc_len': dev_doc_len
        }
        self.test_dataset = {
            'doc_id': te_doc_id,
            'y_emotion': te_y_emotion,
            'y_cause': te_y_cause,
            'y_pairs': te_y_pairs,
            'x': te_x,
            'clause_len': te_clause_len,
            'doc_len': te_doc_len
        }

        # 获取 word embedding
        # 下图需要时 tensor
        self.model = getattr(models, self.args.model_name)(self.args,
                                                           torch.from_numpy(
                                                               self.word_embedding_lookup_table).float().to(
                                                               self.args.device))
        self.model.to(self.args.device)
        optimizer_parameters = [
            {
                'params': self.model.parameters(),
                'weight_decay': self.args.weight_decay
            }
        ]

        self.optimizer = torch.optim.Adam(optimizer_parameters, lr=self.args.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # 不需要使用 scheduler
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.args.learning_rate)

        self.best_f1 = -1
        self.emotion_f1 = -1
        self.cause_f1 = -1

        self.train_time = 0
        self.dev_time = 0
        self.test_time = 0
        # 要返回的 train_time 的值
        self.return_train_time = 0
        self.return_dev_time = 0
        self.return_test_time = 0

    def train(self):
        self.model.zero_grad()
        for epoch in range(self.args.num_epoch):
            logger.info(f'第 {epoch} 个 epoch')
            self.__train_one_epoch(in_which_epoch=epoch)
            emotion_f1, cause_f1, cur_f1 = self.eval(eval_dataset='test')
            if cur_f1 > self.best_f1:
                logger.info(f'当前最大的 f1 值跟新了，现在是 {cur_f1}, 原来是 {self.best_f1}')
                self.best_f1 = cur_f1
                self.emotion_f1 = emotion_f1
                self.cause_f1 = cause_f1

                self.eval(eval_dataset='train', write_file=True)
                self.eval(eval_dataset='dev', write_file=True)
                self.eval(eval_dataset='test', write_file=True)
                self.return_train_time = self.train_time
                self.return_dev_time = self.dev_time
                self.return_test_time = self.test_time
        logger.info('训练过程结束了')
        logger.info(f'emotion f1 is : {self.emotion_f1}, cause f1 is : {self.cause_f1}, best f1 is : {self.best_f1}')
        return self.return_train_time, self.return_dev_time, self.return_test_time

    def __train_one_epoch(self, in_which_epoch=0):
        self.model.train()
        start_time = time.time()
        inputs = {
            'x': torch.from_numpy(self.train_dataset['x']).long().to(self.args.device),
            'clause_len': self.train_dataset['clause_len'],
            'doc_len': self.train_dataset['doc_len'],
            'y_emotion': torch.from_numpy(self.train_dataset['y_emotion']).long().to(self.args.device),
            'y_cause': torch.from_numpy(self.train_dataset['y_cause']).long().to(self.args.device),
            'batch_size': self.args.batch_size,
            'test': False
        }
        batch_count = 0
        for batch_input, _ in get_batch_data(**inputs):
            # batch_input 是 inputs 的前五项 （一个 tuple
            predict_emotion, predict_cause = self.model(batch_input[0])

            emotion_label = batch_input[3]
            cause_label = batch_input[4]

            emotion_loss = self.criterion(predict_emotion.permute(0, 2, 1), emotion_label)
            cause_loss = self.criterion(predict_cause.permute(0, 2, 1), cause_label)

            loss = emotion_loss + cause_loss

            # 标准三步走
            loss.backward()
            self.optimizer.step()
            # 清空梯度之前一定要 先优化
            self.model.zero_grad()

            batch_count += 1

            # emotion_acc, emotion_precision, emotion_recall, emotion_f1 = compute_acc_precision_recall_f1(
            #     predict_emotion, emotion_label, batch_input[2])
            # cause_acc, cause_precision, cause_recall, cause_f1 = compute_acc_precision_recall_f1(
            #     predict_cause,
            #     cause_label,
            #     batch_input[2]
            # )
            # logger.info(emotion_f1)
            # logger.info(cause_f1)
        end_time = time.time()
        total_time = end_time - start_time
        self.train_time = batch_count / total_time

    def eval(self, eval_dataset='test', write_file=False):
        self.model.eval()
        if eval_dataset == 'dev':
            dataset = self.dev_dataset
        elif eval_dataset == 'test':
            dataset = self.test_dataset
        elif eval_dataset == 'train':
            dataset = self.train_dataset
        else:
            logger.warning('默认应该是test dataset')
            dataset = self.test_dataset
        start_time = time.time()
        inputs = {
            'x': torch.from_numpy(dataset['x']).long().to(self.args.device),
            'clause_len': dataset['clause_len'],
            'doc_len': dataset['doc_len'],
            'y_emotion': torch.from_numpy(dataset['y_emotion']).long().to(self.args.device),
            'y_cause': torch.from_numpy(dataset['y_cause']).long().to(self.args.device),
            'batch_size': self.args.batch_size,
            'test': False
        }
        emotion_predict_label = None
        cause_predict_label = None
        emotion_true_label = None
        cause_true_label = None

        batch_count = 0
        for batch_input, _ in get_batch_data(**inputs):
            # batch_input 是 inputs 的前五项 （一个 tuple
            with torch.no_grad():
                predict_emotion, predict_cause = self.model(batch_input[0])

            batch_count += 1
            # emotion_label = batch_input[3]
            # cause_label = batch_input[4]

            # emotion_loss = self.criterion(predict_emotion.permute(0, 2, 1), emotion_label)
            # cause_loss = self.criterion(predict_cause.permute(0, 2, 1), cause_label)
            if emotion_predict_label is None:
                emotion_predict_label = predict_emotion.detach().cpu().numpy()
                emotion_true_label = batch_input[3].detach().cpu().numpy()
                cause_predict_label = predict_cause.detach().cpu().numpy()
                cause_true_label = batch_input[4].detach().cpu().numpy()
            else:
                emotion_predict_label = np.append(emotion_predict_label,
                                                  predict_emotion.detach().cpu().numpy(), axis=0)
                emotion_true_label = np.append(emotion_true_label,
                                               batch_input[3].detach().cpu().numpy(), axis=0)
                cause_predict_label = np.append(cause_predict_label,
                                                predict_cause.detach().cpu().numpy(), axis=0)
                cause_true_label = np.append(cause_true_label,
                                             batch_input[4].detach().cpu().numpy(), axis=0)

        end_time = time.time()
        total_time = end_time - start_time
        if eval_dataset == 'dev':
            self.dev_time = batch_count / total_time
        elif eval_dataset == 'test':
            self.test_time = batch_count / total_time

        emotion_acc, emotion_precision, emotion_recall, emotion_f1 = compute_acc_precision_recall_f1(
            emotion_predict_label, emotion_true_label, inputs['doc_len'])
        cause_acc, cause_precision, cause_recall, cause_f1 = compute_acc_precision_recall_f1(
            cause_predict_label,
            cause_true_label,
            inputs['doc_len']
        )
        logger.info(f'当前正在测试 ：{eval_dataset} 数据集')
        logger.info(f'emotion_f1 is : {emotion_f1}, cause f1 is : {cause_f1}')

        if write_file:
            self.get_pair_data(f'./inputs/tmp/{eval_dataset}.txt',
                               emotion_predict_label, cause_predict_label,
                               train=eval_dataset)

        return emotion_f1, cause_f1, (emotion_f1 + cause_f1) / 2

    def get_pair_data(self, file_name, pred_y_emotion, pred_y_cause, train='train'):
        # pred_y_emotion shape is ( batch, doc_len, emotion )
        if train == 'train':
            data = self.train_dataset
        elif train == 'dev':
            data = self.dev_dataset
        else:
            data = self.test_dataset
        pred_y_cause = pred_y_cause.argmax(2)
        pred_y_emotion = pred_y_emotion.argmax(2)
        # if train == 'dev':
        #     print(pred_y_cause)
        #     print(pred_y_emotion.shape)
        #     cnt = 0
        #     for i in pred_y_emotion:
        #         if 1 in i:
        #             cnt += 1
        #     print(cnt)
        #     exit(1)
        f = open(file_name, 'w')
        # 这个其实没有什么用
        self.idx2word[0] = '<UNK>'
        for i in range(len(data['doc_id'])):
            f.write(data['doc_id'][i] + ' ' + str(data['doc_len'][i]) + '\n')
            f.write(str(data['y_pairs'][i]) + '\n')
            for j in range(data['doc_len'][i]):
                clause = ''
                for k in range(data['clause_len'][i][j]):
                    clause = clause + self.idx2word[data['x'][i][j][k]] + ' '

                f.write(
                    str(j + 1) + ', ' + str(pred_y_emotion[i][j]) + ', ' + str(
                        pred_y_cause[i][j]) + ', ' + clause + '\n')
        f.close()
        print('write {} done'.format(file_name))
