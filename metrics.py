from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def compute_acc_precision_recall_f1(predict_logits, golden_labels, doc_len, average='binary'):
    # predict_logits shape is (batch, doc_len, num_class)
    # golden labels shape is (batch, doc_len)
    # doc_len shape is (batch)
    predict_label = predict_logits.argmax(2)
    predicts = []
    goldens = []
    for predict, golden, one_doc_len in zip(predict_label, golden_labels, doc_len):
        for i in range(one_doc_len):
            predicts.append(predict[i])
            goldens.append(golden[i])

    predicts = [x.item() for x in predicts]
    goldens = [x.item() for x in goldens]
    # print(classification_report(
    #     goldens,
    #     predicts,
    #     digits=4
    # ))
    acc = precision_score(goldens, predicts, average='micro')
    p = precision_score(goldens, predicts, average=average)
    r = recall_score(goldens, predicts, average=average)
    f1 = f1_score(goldens, predicts, average=average)
    return acc, p, r, f1


def compute_pair_precision_recall_f1(pair_id_all, pair_id, pred_y, fold=0, save_dir=''):
    pair_id_filtered = []

    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_filtered.append(pair_id[i])

    def write_log():
        pair_to_y = dict(zip(pair_id, pred_y))
        g = open(save_dir + 'pair_log_fold{}.txt'.format(fold), 'w')
        doc_id_b, doc_id_e = pair_id_all[0] / 10000, pair_id_all[-1] / 10000
        idx_1, idx_2 = 0, 0
        for doc_id in range(doc_id_b, doc_id_e + 1):
            true_pair, pred_pair, pair_y = [], [], []
            line = str(doc_id) + ' '
            while True:
                p_id = pair_id_all[idx_1]
                d, p1, p2 = p_id / 10000, p_id % 10000 / 100, p_id % 100
                if d != doc_id: break
                true_pair.append((p1, p2))
                line += '({}, {}) '.format(p1, p2)
                idx_1 += 1
                if idx_1 == len(pair_id_all):
                    break
            line += '|| '
            while True:
                p_id = pair_id[idx_2]
                d, p1, p2 = p_id / 10000, p_id % 10000 / 100, p_id % 100
                if d != doc_id:
                    break
                if pred_y[idx_2]:
                    pred_pair.append((p1, p2))
                pair_y.append(pred_y[idx_2])
                line += '({}, {}) {} '.format(p1, p2, pred_y[idx_2])
                idx_2 += 1
                if idx_2 == len(pair_id): break
            if len(true_pair) > 1:
                line += 'multipair '
                if true_pair == pred_pair:
                    line += 'good '
            line += '\n'
            g.write(line)

    if fold:
        write_log()
    keep_rate = len(pair_id_filtered) / (len(pair_id) + 1e-8)
    s1, s2, s3 = set(pair_id_all), set(pair_id), set(pair_id_filtered)
    # 没有预测到的
    o_acc_num = len(s1 & s2)
    # 预测对了的
    acc_num = len(s1 & s3)
    o_p, o_r = o_acc_num / (len(s2) + 1e-8), o_acc_num / (len(s1) + 1e-8)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1, o_f1 = 2 * p * r / (p + r + 1e-8), 2 * o_p * o_r / (o_p + o_r + 1e-8)

    return p, r, f1, o_p, o_r, o_f1, keep_rate


def main():
    pass


if __name__ == '__main__':
    main()
