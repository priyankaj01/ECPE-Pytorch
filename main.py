# 一些常见的自带的库
import os
import random
import numpy as np

# pytorch
import torch
import torch.cuda
import torch.backends.cudnn

# 个人实现
from args import args
import trainers

# 文件log系统
import logging
import logging.config
# import mylog  # 使用自己配置的 log 系统
# TODO: 到时候需要换成好一点的实现
from mylog import LOG_CONFIG


def set_log():
    logging.config.dictConfig(LOG_CONFIG)
    logger = logging.getLogger(__name__)
    return logger


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    # 设置 log 系统
    logger = set_log()
    # 获取所有的参数

    # 设置随机数种子
    # set_seed(args)

    # 进行模型的训练
    # 根据参数选择 trainer
    Trainer = getattr(trainers, args.trainer_name)
    trainer = Trainer(args)
    tr_time, de_time, te_time = 0, 0, 0
    if args.do_train:
        tr_time, de_time, te_time = trainer.train()

    # 在做 test 之前，应该需要 load 以前保存的最佳模型
    if args.do_test:
        trainer.test()
    return tr_time, de_time, te_time


def compute_time():
    # 不需要看这个
    train_time, dev_time, test_time = [], [], []
    pair_train_time, pair_dev_time, pair_test_time = [], [], []
    args.num_epoch = 20
    args.batch_size = 3
    args.all_data_file = './inputs/all_data_fc.txt'
    for i in range(10):
        args.model_name = 'IndependentBiLSTM'
        args.trainer_name = 'BaseTrainer'
        args.train_file_path = '/home/yuanchaofa/DataSplits/split_{}/train.txt'.format(i + 1)
        args.dev_file_path = '/home/yuanchaofa/DataSplits/split_{}/dev.txt'.format(i + 1)
        args.test_file_path = '/home/yuanchaofa/DataSplits/split_{}/test.txt'.format(i + 1)
        ts = main()
        train_time.append(ts[0])
        dev_time.append(ts[1])
        test_time.append(ts[2])

        args.model_name = 'ConstructPairs'
        args.trainer_name = 'ConstructPairsTrainer'
        args.train_file_path = './inputs/tmp/train.txt'.format(i + 1)
        args.dev_file_path = './inputs/tmp/dev.txt'.format(i + 1)
        args.test_file_path = './inputs/tmp/test.txt'.format(i + 1)
        ts = main()
        pair_train_time.append(ts[0])
        pair_dev_time.append(ts[1])
        pair_test_time.append(ts[2])
    with open('time_log.txt', 'w') as f:
        f.write(str(train_time))
        f.write('\n')
        f.write(str(pair_train_time))
        f.write('\n')
        f.write(str(dev_time))
        f.write('\n')
        f.write(str(pair_dev_time))
        f.write('\n')
        f.write(str(test_time))
        f.write('\n')
        f.write(str(pair_test_time))

    train_time = (sum(train_time) + sum(pair_train_time)) / 20
    dev_time = (sum(dev_time) + sum(pair_dev_time)) / 20
    test_time = (sum(test_time) + sum(pair_test_time)) / 20

    print('train time is: {}'.format(train_time))
    print('dev time is: {}'.format(dev_time))
    print('test time is: {}'.format(test_time))




if __name__ == '__main__':
    main()
    # compute_time()