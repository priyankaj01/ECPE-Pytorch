import torch
import torch.nn as nn
import torch.nn.functional as F

label2id = {'C_-4': 0, 'C_0': 1, 'C_2': 2, 'C_-3': 3, 'O': 4,
            'C_1': 5, 'C_-2': 6, 'C_4': 7, 'C_3': 8, 'C_-1': 9, 'E': 10}

# labels = ['C_-4', 'C_0', 'C_2', 'C_-3', 'O',
#           'C_1', 'C_-2', 'C_4', 'C_3', 'C_-1']
#
#
# def get_pos_reg(keys, doc_len):
#     res_list = []
#     for key in keys:
#         if key.startswith('C'):
#             distance = int(key.split('_')[-1])
#             res = 1 - abs(abs(distance) - 1) / doc_len
#         else:
#             res = 0
#         res_list.append(res)
#     return res_list


def pos_reg(doc_lens, logits):
    labels = ['C_-4', 'C_0', 'C_2', 'C_-3', 'O',
              'C_1', 'C_-2', 'C_4', 'C_3', 'C_-1']

    def get_pos_reg(keys, one_doc_len):
        res_list = []
        for key in keys:
            if key.startswith('C'):
                distance = int(key.split('_')[-1])
                res = 1 - abs(abs(distance) - 1) / one_doc_len
            else:
                res = 0
            res_list.append(res)
        return res_list

    tmp = []
    start = 0
    for doc_len in doc_lens:
        pos_logits = get_pos_reg(labels, doc_len)
        pos_logits = torch.tensor(pos_logits)
        # pos_logits = torch.tensor(pos_logits).to(opt.device)
        end = start + doc_len
        tmp.append(logits[start:end, :] * pos_logits)
        start = end
    loss = torch.sum(
        torch.cat(tmp, dim=0)
    )
    return loss


def get_relative_position(n):
    res = [[i for i in range(n)]]
    for i in range(1, n):
        res.append([x - i for x in res[0]])
    res = torch.tensor(res, dtype=torch.long)
    res = res + 81
    return res


def main():
    doc_lens = [5, 6, 7, 8]
    logits = torch.rand(5 + 6 + 7 + 8, 10)
    print(pos_reg(doc_lens, logits))


if __name__ == '__main__':
    main()
