"""
Construct pairs trainer
"""
import models
import torch.optim
import torch.nn as nn
import logging
import numpy as np
import time

from data.prepare_data import load_word2vector, load_data_for_pair
from data.get_batch_data import get_batch_data_for_pair
from utils.uitls import convert_one_hot_to_index
from metrics import compute_pair_precision_recall_f1

logger = logging.getLogger(__name__)


class ConstructPairsTrainer(object):
    def __init__(self, args):
        super(ConstructPairsTrainer, self).__init__()
        self.args = args
        self.idx2word, self.word2idx, self.word_embedding_lookup_table, self.position_embedding_lookup_table = load_word2vector(
            self.args.embedding_dim,
            self.args.embedding_dim_pos,
            self.args.all_data_file,
            self.args.w2v_file
        )

        tr_pair_id_all, tr_pair_id, tr_y, tr_x, tr_clause_len, tr_distance = load_data_for_pair(
            self.args.train_file_path, self.word2idx,
            self.args.max_doc_len, self.args.max_clause_len)
        dev_pair_id_all, dev_pair_id, dev_y, dev_x, dev_clause_len, dev_distance = load_data_for_pair(
            self.args.test_file_path, self.word2idx,
            self.args.max_doc_len, self.args.max_clause_len)
        te_pair_id_all, te_pair_id, te_y, te_x, te_clause_len, te_distance = load_data_for_pair(
            self.args.test_file_path, self.word2idx,
            self.args.max_doc_len, self.args.max_clause_len)

        self.train_dataset = {
            'pair_id_all': tr_pair_id_all,
            'pair_id': tr_pair_id,
            'y': tr_y,
            'x': tr_x,
            'clause_len': tr_clause_len,
            'distance': tr_distance,
        }
        self.dev_dataset = {
            'pair_id_all': dev_pair_id_all,
            'pair_id': dev_pair_id,
            'y': dev_y,
            'x': dev_x,
            'clause_len': dev_clause_len,
            'distance': dev_distance,
        }
        self.test_dataset = {
            'pair_id_all': te_pair_id_all,
            'pair_id': te_pair_id,
            'y': te_y,
            'x': te_x,
            'clause_len': te_clause_len,
            'distance': te_distance,
        }

        # 获取 word embedding
        # 下图需要时 tensor
        self.model = getattr(models, self.args.model_name)(self.args,
                                                           torch.from_numpy(
                                                               self.word_embedding_lookup_table
                                                           ).float().to(
                                                               self.args.device),
                                                           torch.from_numpy(
                                                               self.position_embedding_lookup_table
                                                           ).float().to(
                                                               self.args.device
                                                           ))
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

        self.best_f1 = 0
        self.train_time = 0
        self.test_time = 0
        self.dev_time = 0

        # 要返回的 train_time 的值
        self.return_train_time = 0
        self.return_dev_time = 0
        self.return_test_time = 0

    def train(self):
        self.model.zero_grad()
        for epoch in range(self.args.num_epoch):
            logger.info(f'第 {epoch} 个 epoch')
            self.__train_one_epoch(in_which_epoch=epoch)
            cur_f1 = self.eval(eval_dataset='test')
            if cur_f1 > self.best_f1:
                self.best_f1 = cur_f1
                self.eval(eval_dataset='test')
                self.return_train_time = self.train_time
                self.return_dev_time = self.dev_time
                self.return_test_time = self.test_time
        return self.return_train_time, self.return_dev_time, self.return_test_time

    def __train_one_epoch(self, in_which_epoch=0):
        self.model.train()
        start_time = time.time()
        inputs = {
            'x': torch.from_numpy(self.train_dataset['x']).long().to(self.args.device),
            'y': torch.from_numpy(self.train_dataset['y']).long().to(self.args.device),
            'distance': torch.from_numpy(self.train_dataset['distance']).long().to(self.args.device),
            'clause_len': torch.from_numpy(self.train_dataset['clause_len']).long().to(self.args.device),
            'batch_size': self.args.batch_size,
            'test': False
        }
        batch_cnt = 0
        for batch_input, _ in get_batch_data_for_pair(**inputs):
            # batch_input 是 inputs 的前五项 （一个 tuple
            predicts = self.model(batch_input[0], batch_input[2])
            golden_label = batch_input[1]

            loss = self.criterion(predicts, golden_label)

            # 标准三步走
            loss.backward()
            self.optimizer.step()
            # 清空梯度之前一定要 先优化
            self.model.zero_grad()

            batch_cnt += 1
            # 训练集也可以进行一下计算 acc, r, p, f1

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
        self.train_time = batch_cnt / total_time

    def eval(self, eval_dataset='test'):
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
            'y': torch.from_numpy(dataset['y']).long().to(self.args.device),
            'distance': torch.from_numpy(dataset['distance']).long().to(self.args.device),
            'clause_len': torch.from_numpy(dataset['clause_len']).long().to(self.args.device),
            'batch_size': self.args.batch_size,
            'test': False
        }

        predict_label = None
        # golden_label = None

        self.model.eval()

        batch_cnt = 0
        for batch_input, _ in get_batch_data_for_pair(**inputs):
            # batch_input 是 inputs 的前五项 （一个 tuple
            with torch.no_grad():
                predicts = self.model(batch_input[0], batch_input[2])
            # tmp_golden_label = batch_input[1]

            batch_cnt += 1
            if predict_label is None:
                predict_label = predicts.detach().cpu().numpy()
                # golden_label = tmp_golden_label.detach().cpu().numpy()
            else:
                predict_label = np.append(predict_label, predicts.detach().cpu().numpy(), axis=0)
                # golden_label = np.append(golden_label, tmp_golden_label.detach().cpu().numpy(), axis=1)

        end_time = time.time()
        total_time = end_time - start_time

        if eval_dataset == 'dev':
            self.dev_time = batch_cnt / total_time
        elif eval_dataset == 'test':
            self.test_time = batch_cnt / total_time
        precision, recall, f1, _, _, filtered_f1, keep_rate = compute_pair_precision_recall_f1(
            self.test_dataset['pair_id_all'],
            self.test_dataset['pair_id'],
            predict_label.argmax(-1)
        )
        logger.info(f'test f1 is : {f1} , and filtered f1 is : {filtered_f1}')
        logger.info(f'logistic regression keep rate is: {keep_rate}')
        return f1
