import numpy as np

import logging

logger = logging.getLogger(__name__)


def load_word2vector(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    logger.info('加载词向量')
    words = []
    with open(train_file_path, 'r', encoding = 'utf-8') as f1:
        for line in f1.readlines():
            line = line.strip()
            words.append(line)
        words = set(words)  # redupliction removing
        word2idx = dict((c, k + 1) for k, c in enumerate(words))  # each word and its position
        idx2word = dict((k + 1, c) for k, c in enumerate(words))
    w2v = {}
    with open(embedding_path, 'r', encoding='utf-8') as f2:
        # embedding file path, 里面包含了很多次，比预想的要多，实际上只需要其中的一部分
        f2.readline()  # Q read the first line: 43593 200
        for line in f2.readlines():
            line = line.strip().split(' ')
            w, ebd = line[0], line[1:]
            w2v[w] = ebd

    # 初始化了 0 的 embedding
    embedding = [list(np.zeros(embedding_dim))]

    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            # 不同的词采用不同的随机初始化
            # TODO: 这里其实可以采用 统一的 <UNK> 表示
            vec = list(
                np.random.rand(embedding_dim) / 5. - 0.1)  # take randomly from the uniform distribution[-0.1, 0.1]
        embedding.append(vec)
    logger.info('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    # 初始化 position embedding
    embedding_pos = [list(np.zeros(embedding_dim_pos))]  # Q2
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)])
    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    logger.info('embedding.shape: {} embedding_pos.shape: {}'.format(embedding.shape, embedding_pos.shape))
    logger.info('load embedding done!\n')
    return idx2word, word2idx, embedding, embedding_pos


def load_word2vector_(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    # train file path 其实是 包含了 Train， dev test 的所有样本
    # 是在 input 中的 clause keywords.csv中
    # 注意这里得embedding dim 指的是 word2vec
    # embedding_dim_pos 指的是 position 的 embedding dim
    logger.info('load embedding...')
    logger.info(train_file_path)

    # 存储所有的 word
    words = []
    with open(train_file_path, 'r') as f1:
        for line in f1.readlines():
            line = line.strip().split(',')
            emotion, clause = line[2], line[-1]
            words.extend([emotion] + clause.split())
        words = set(words)  # redupliction removing
        word2idx = dict((c, k + 1) for k, c in enumerate(words))  # each word and its position
        idx2word = dict((k + 1, c) for k, c in enumerate(words))

    w2v = {}
    with open(embedding_path, 'r') as f2:
        # embedding file path, 里面包含了很多次，比预想的要多，实际上只需要其中的一部分
        f2.readline()  # Q read the first line: 43593 200
        for line in f2.readlines():
            line = line.strip().split(' ')
            w, ebd = line[0], line[1:]
            w2v[w] = ebd

    # 初始化了 0 的 embedding
    embedding = [list(np.zeros(embedding_dim))]

    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            # 不同的词采用不同的随机初始化
            # TODO: 这里其实可以采用 统一的 <UNK> 表示
            vec = list(
                np.random.rand(embedding_dim) / 5. - 0.1)  # take randomly from the uniform distribution[-0.1, 0.1]
        embedding.append(vec)
    logger.info('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    # 初始化 position embedding
    embedding_pos = [list(np.zeros(embedding_dim_pos))]  # Q2
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)])
    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    logger.info('embedding.shape: {} embedding_pos.shape: {}'.format(embedding.shape, embedding_pos.shape))
    logger.info('load embedding done!\n')
    return idx2word, word2idx, embedding, embedding_pos


# 用于 加载 训练集/test dataset 的的函数
def load_data(input_file, word_idx, max_doc_len=75, max_clause_len=45):
    logger.info('load data_file: {}'.format(input_file))
    # y_position, y_cause, y_pairs, x, sen_len, doc_len = [], [], [], [], [], []
    y_emotion, y_cause, y_pairs, x, clause_len, doc_len = [], [], [], [], [], []
    doc_id = []

    # 记录有多少被 doc 被切除了 一些子句
    n_cut = 0
    with open(input_file, 'r', encoding='utf-8') as f1:
        while True:
            line = f1.readline()
            if line == '':
                break
            line = line.strip().split()
            doc_id.append(line[0])  # 212 10
            d_len = int(line[1])
            pairs = eval('[' + f1.readline().strip() + ']')
            doc_len.append(d_len)
            y_pairs.append(pairs)
            # print(pairs)
            # exit(1)
            pos, cause = zip(*pairs)
            # 下面的 pos 其实指的是 emotion

            # 下面的输入 不符合 pytorch 的使用，所以做了修改
            y_emotion_tmp, y_cause_tmp, clause_len_tmp, x_tmp = np.zeros(max_doc_len), np.zeros(
                max_doc_len), np.zeros(max_doc_len, dtype=np.int32), np.zeros((max_doc_len, max_clause_len),
                                                                              dtype=np.int32)
            for i in range(d_len):
                if i + 1 in pos:
                    y_emotion_tmp[i] = 1
                if i + 1 in cause:
                    y_cause_tmp[i] = 1
                words = f1.readline().strip().split(',')[-1]  # get clause
                clause_len_tmp[i] = min(len(words.split()), max_clause_len)
                for j, word in enumerate(words.split()):
                    if j >= max_clause_len:
                        n_cut += 1
                        break
                    try:
                        x_tmp[i][j] = int(word_idx[word])
                    except:
                        # 这里写一个 try 实际上不会对用户有影响
                        x_tmp[i][j] = 0

            y_emotion.append(y_emotion_tmp)
            y_cause.append(y_cause_tmp)
            x.append(x_tmp)
            clause_len.append(clause_len_tmp)

        y_emotion, y_cause, x, clause_len, doc_len = map(np.array, [y_emotion, y_cause, x, clause_len, doc_len])
        for var in ['y_emotion', 'y_cause', 'x', 'clause_len', 'doc_len']:
            logger.info('{}.shape: {}'.format(var, eval(var).shape))
        logger.info('被切分了的 clause: {}'.format(n_cut))
        logger.info('load data done!\n')
        return doc_id, y_emotion, y_cause, y_pairs, x, clause_len, doc_len


def load_data_for_pair(input_file, word_idx, max_doc_len=75, max_clause_len=45):
    logger.info('load data_file for pair: {}'.format(input_file))
    pair_id_all, pair_id, y, x, sen_len, distance = [], [], [], [], [], []

    n_cut = 0
    inputFile = open(input_file, 'r')
    while True:
        line = inputFile.readline()
        if line == '':
            break
        line = line.strip().split()
        doc_id = int(line[0])
        d_len = int(line[1])
        pairs = eval(inputFile.readline().strip())
        try:
            pair_id_all.extend([doc_id * 10000 + p[0] * 100 + p[1] for p in pairs])  # eg. 3 15 (12,12) -> 3|12|12
        except:
            pair_id_all.extend(
                [doc_id * 10000 + int(pairs[0]) * 100 + int(pairs[1])])  # eg. 3 15 (12,12) -> 3|12|12

        # pair_id_all.extend([doc_id * 10000 + p[0] * 100 + p[1] for p in pairs])
        sen_len_tmp, x_tmp = np.zeros(max_doc_len, dtype=np.int32), np.zeros((max_doc_len, max_clause_len),
                                                                             dtype=np.int32)
        pos_list, cause_list = [], []
        for i in range(d_len):
            line = inputFile.readline().strip().split(',')
            if int(line[1].strip()) > 0:
                pos_list.append(i + 1)
            if int(line[2].strip()) > 0:
                cause_list.append(i + 1)
            words = line[-1]
            sen_len_tmp[i] = min(len(words.split()), max_clause_len)
            for j, word in enumerate(words.split()):
                if j >= max_clause_len:
                    n_cut += 1
                    break
                try:
                    x_tmp[i][j] = int(word_idx[word])
                except:
                    x_tmp[i][j] = 0
        for i in pos_list:
            for j in cause_list:
                pair_id_cur = doc_id * 10000 + i * 100 + j
                pair_id.append(pair_id_cur)
                y.append(1 if pair_id_cur in pair_id_all else 0)
                # y.append([0, 1] if pair_id_cur in pair_id_all else [1, 0])
                x.append([x_tmp[i - 1], x_tmp[j - 1]])
                sen_len.append([sen_len_tmp[i - 1], sen_len_tmp[j - 1]])
                distance.append(j - i + 100)
    y, x, sen_len, distance = map(np.array, [y, x, sen_len, distance])
    for var in ['y', 'x', 'sen_len', 'distance']:
        logger.info('{}.shape {}'.format(var, eval(var).shape))
    logger.info('n_cut {}, (y-negative, y-positive): {}'.format(n_cut, y.sum(axis=0)))
    logger.info('load data done!\n')
    return pair_id_all, pair_id, y, x, sen_len, distance


def main():
    import logging.config
    from mylog import LOG_CONFIG
    logging.config.dictConfig(LOG_CONFIG)

    embedding_dim = 200
    embedding_dim_pos = 50
    train_file_path = '../inputs/clause_keywords.csv'
    embedding_path = '../inputs/w2v_200.txt'

    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_word2vector(embedding_dim,
                                                                                    embedding_dim_pos,
                                                                                    train_file_path,
                                                                                    embedding_path)
    s = load_data_for_pair('../inputs/test_input/fold1_test.txt',
                           word_id_mapping)
    print(len(s[0]))
    print(len(s[1]))
    print(s[2])
    print(len(s[2]))
    print(s[3].shape)

    exit(1)


if __name__ == '__main__':
    main()
