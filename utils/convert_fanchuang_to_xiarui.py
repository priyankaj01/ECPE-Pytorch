import json
import pickle
import os


def get_all_emotion_cause(pairs, doc_len):
    # 实际上不需要这个也可以
    emotion_list = [0] * doc_len
    cause_list = [0] * doc_len
    emotions = set()
    causes = set()
    for pair in pairs:
        emotions.add(pair[0])
        causes.add(pair[1])

    for i in range(doc_len):
        if i in emotions:
            emotion_list[i] = 1
        if i in causes:
            cause_list[i] = 1
    return emotion_list, cause_list


def convert_fc_to_xr(file_name, data, all_dict):
    # 下面的代码让我自我怀疑
    f = open(file_name, 'w')
    doc_list = data[0]
    pairs = data[1]
    for i in range(len(doc_list)):
        all_clauses = all_dict[doc_list[i]]
        f.write(str(i + 1))
        f.write(' ')
        f.write(str(len(doc_list[i].split('\x01'))))
        f.write('\n')
        for idx, pair in enumerate(pairs[i]):
            if idx == 0:
                f.write(' ')
            else:
                f.write(', ')
            pair = (pair[0] + 1, pair[1] + 1)
            f.write(str(pair))
        f.write('\n')
        for idx, clause in enumerate(all_clauses['text'].split('\x01')):
            f.write(str(idx + 1))
            f.write(',0,0,')
            f.write(clause)
            f.write('\n')
    f.close()


def main():
    all_data = '/home/yuanchaofa/exp_data/pair_extraction/input/cross_validation/fold_9'
    train_dict = json.load(open(all_data + '/train.json', 'r'))
    test_dict = json.load(open(all_data + '/test.json', 'r'))
    all_dict = {**train_dict, **test_dict}
    # f = open('../inputs/all_data_fc.txt', 'w')
    # for key, value in all_dict.items():
    #     clauses = value['text']
    #     for clause in clauses.split('\x01'):
    #         for word in clause.split(' '):
    #             f.write(word)
    #             f.write('\n')
    # f.close()
    # exit(1)
    for i in range(1, 36):
        # for i in [4, 26, 35, 27, 10, 9, 28, 20, 31, 16, 32, 19, 6, 22, 2, 23, 5, 7, 24, 13]:
        file_dir = f'/data10T/yuanchaofa/pair_ml/input/fold_{i}'
        save_dir = '/data10T/yuanchaofa/DataSplits'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # file_dir = file_dir + f'/split_{i}'
        train_data = pickle.load(open(file_dir + '/train.pkl', 'rb'))
        dev_data = pickle.load(open(file_dir + '/valid.pkl', 'rb'))
        test_data = pickle.load(open(file_dir + '/test.pkl', 'rb'))

        if not os.path.exists(save_dir + f'/fold_{i}'):
            os.mkdir(save_dir + f'/fold_{i}')
        convert_fc_to_xr(save_dir + f'/fold_{i}/train.txt', train_data, all_dict)
        convert_fc_to_xr(save_dir + f'/fold_{i}/dev.txt', dev_data, all_dict)
        convert_fc_to_xr(save_dir + f'/fold_{i}/test.txt', test_data, all_dict)


if __name__ == '__main__':
    main()
