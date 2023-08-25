import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from transformers import BertModel, BertPreTrainedModel

from config import opt


class BiLSTMConfig(object):
    bidirectional = True
    direction_nums = 2
    num_layers = 1
    dropout = 0.5
    batch_first = True


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.key_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def self_attention(self, query, key, value,
                       mask=None, hidden_size=None,
                       project_size=None, dropout=None):
        # 对于Q, K, V 的 shape 都是 (batch, seq_len, hidden_dim)
        # todo:: 了解 hidden_size, and project_size
        # key, query, value = self.get_key_query_embedding(
        #     key, query, value,
        #     hidden_size=hidden_size,
        #     project_size=project_size)
        # Compute 'Scaled Dot Product Attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # 把要 mask，也就是说 mask 值为 0 的，变成负无穷
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # return torch.matmul(p_attn, value), p_attn
        return torch.matmul(p_attn, value)

    def get_key_query_embedding(self, key, query, value, hidden_size, project_size):
        # 对 key, query, value 三个指进行映射
        key = self.key_layer(key)
        query = self.query_layer(query)
        value = self.value_layer(value)
        return key, query, value


class BertClsMulti(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClsMulti, self).__init__(config)

        self.bert = BertModel(config)
        self.num_labels = opt.num_labels

        config.hidden_state_size = 512
        self.lstm_direction_nums = BiLSTMConfig.direction_nums if BiLSTMConfig.bidirectional else 1
        self.clause_lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_state_size // self.lstm_direction_nums,
            # hidden_size=256//self.lstm_direction_nums,
            bidirectional=BiLSTMConfig.bidirectional,
            batch_first=BiLSTMConfig.batch_first
        )
        # 使用 512 就是 0.636
        self.mlp_size = 256
        self.dropout = 0.5
        self.scale_factor = 2
        self.MLP = nn.Sequential(
            nn.Linear(config.hidden_state_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
            # nn.Linear(self.mlp_size, self.mlp_size//self.scale_factor),
            # nn.BatchNorm1d(self.mlp_size//self.scale_factor),
            # nn.LeakyReLU(),
            # nn.Dropout(self.dropout),
            # nn.Linear(self.mlp_size//self.scale_factor, self.num_labels)
        )
        # self.emotion_mlp = nn.Linear(self.mlp_size, 2)
        # self.cause_mlp = nn.Linear(self.mlp_size, 2)
        # self.emotion_mlp = nn.Linear(config.hidden_state_size, 2)
        # self.cause_mlp = nn.Linear(config.hidden_state_size, 2)

        self.emotion_mlp = nn.Sequential(
            nn.Linear(config.hidden_state_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            # nn.Linear(self.mlp_size, self.mlp_size//self.scale_factor),
            # nn.BatchNorm1d(self.mlp_size//self.scale_factor),
            # nn.LeakyReLU(),
            # nn.Dropout(self.dropout),
            # nn.Linear(self.mlp_size//self.scale_factor, self.num_labels)
            nn.Linear(self.mlp_size, 2)
        )
        self.cause_mlp = nn.Sequential(
            nn.Linear(config.hidden_state_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            # nn.Linear(self.mlp_size, self.mlp_size//self.scale_factor),
            # nn.BatchNorm1d(self.mlp_size//self.scale_factor),
            # nn.LeakyReLU(),
            # nn.Dropout(self.dropout),
            # nn.Linear(self.mlp_size//self.scale_factor, self.num_labels)
            nn.Linear(self.mlp_size, 2)
        )
        # self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        # self.classifier = nn.Linear(self.mlp_size, self.num_labels)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_state_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size, self.mlp_size // self.scale_factor),
            nn.BatchNorm1d(self.mlp_size // self.scale_factor),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size // self.scale_factor, self.num_labels)
        )

    def padding_and_mask(self, ids_list):
        max_len = max([len(x) for x in ids_list])
        mask_list = []
        ids_padding_list = []
        # labels_padding_list = []

        for ids in ids_list:
            mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
            ids = ids + [0] * (max_len - len(ids))
            # labels = labels + [0] * (mask_len - len(labels))
            mask_list.append(mask)
            ids_padding_list.append(ids)
            # labels_padding_list.append(labels)
        return ids_padding_list, mask_list

    def forward(self,
                document_list,
                labels,
                emotion_labels,
                cause_labels,
                args,
                tokenizer):

        text_list, tokens_list, ids_list = [], [], []
        document_len = [len(x.split('\x01')) for x in document_list]
        # print(document_list)
        # exit(1)
        for document in document_list:
            text_list.extend(document.strip().split('\x01'))

        for text in text_list:
            text = ''.join(text.split())
            tokens = tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            tokens_list.append(tokens)
        for tokens in tokens_list:
            ids_list.append(tokenizer.convert_tokens_to_ids(tokens))

        ids_padding_list, mask_list = self.padding_and_mask(ids_list)
        ids_padding_tensor = torch.tensor(ids_padding_list, dtype=torch.long).to(args.device)
        attention_mask = torch.tensor(mask_list, dtype=torch.long).to(args.device)

        # print(ids_padding_tensor.shape)
        # exit(1)
        # emotion_labels = None
        # cause_labels = None

        if labels is not None:
            tmp_label = []
            for one_doc_label in labels:
                tmp_label.extend(one_doc_label)
            labels = torch.tensor(tmp_label, dtype=torch.long).to(args.device)

        if emotion_labels is not None:
            tmp_emotion_label = []
            for one_doc_label in emotion_labels:
                tmp_emotion_label.extend(one_doc_label)
            emotion_labels = torch.tensor(tmp_emotion_label, dtype=torch.long).to(args.device)

        if cause_labels is not None:
            tmp_cause_label = []
            for one_doc_label in cause_labels:
                tmp_cause_label.extend(one_doc_label)
            cause_labels = torch.tensor(tmp_cause_label, dtype=torch.long).to(args.device)

        outputs = self.bert(input_ids=ids_padding_tensor, attention_mask=attention_mask)

        pooled = outputs[1]  # batch, hidden_size

        # 对 clause 加一层 lstm
        # highlight TODO
        # 根据 document len 取出不同的 clause
        start = 0
        clauses_rep = []
        for one_doc_len in document_len:
            end = start + one_doc_len
            # clauses_rep.append(self.clause_lstm(pooled[start:end].unsqueeze(0))[0])
            clauses_rep.append(pooled[start:end])
            start = end

        pooled = torch.cat(clauses_rep, dim=1)
        # pooled = pooled.squeeze(0)

        logits = self.classifier(pooled)
        outputs = (logits,)

        if emotion_labels is not None:
            emotion_logits = self.emotion_mlp(pooled)
            outputs = outputs + (emotion_logits,)

        if cause_labels is not None:
            cause_logits = self.cause_mlp(pooled)
            outputs = outputs + (cause_logits,)

        if labels is not None:
            loss_function = nn.CrossEntropyLoss()
            # print('======')
            # print(logits.size(), labels.size())
            pair_loss = loss_function(logits.view(-1, self.num_labels), labels)

        if emotion_labels is not None:
            loss_function = nn.CrossEntropyLoss()
            # print(emotion_logits.size(), emotion_labels.size())
            emotion_loss = loss_function(emotion_logits.view(-1, 2), emotion_labels)
            # outputs = (loss,) + outputs
        if cause_labels is not None:
            loss_function = nn.CrossEntropyLoss()
            cause_loss = loss_function(cause_logits.view(-1, 2), cause_labels)
            # outputs = (loss,) + outputs
        if labels is not None:
            loss = pair_loss

        if cause_labels is not None:
            loss += cause_loss

        if emotion_labels is not None:
            loss += emotion_loss

        if labels is not None:
            outputs = (loss,) + outputs
        return outputs
