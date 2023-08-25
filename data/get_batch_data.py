import numpy as np


def batch_index(length, batch_size, test=False):
    index = list(range(length))
    if not test:
        np.random.shuffle(index)
    for i in range(int((length + batch_size - 1) / batch_size)):
        ret = index[i * batch_size: (i + 1) * batch_size]
        # TODO: 如果 不满一个 batch, 实际上也不会有影响
        # if not test and len(ret) < batch_size:
        #     break
        yield ret


def get_batch_data(x, clause_len, doc_len, y_emotion, y_cause, batch_size, test=False):
    for index in batch_index(len(y_cause), batch_size, test):
        feed_list = [x[index], clause_len[index], doc_len[index], y_emotion[index],
                     y_cause[index]]
        yield feed_list, len(index)


def get_batch_data_for_pair(x, y, distance, clause_len, batch_size, test=False):
    for index in batch_index(len(distance), batch_size, test):
        feed_list = [x[index], y[index], distance[index], clause_len[index]]
        yield feed_list, len(index)


def main():
    pass


if __name__ == '__main__':
    main()
