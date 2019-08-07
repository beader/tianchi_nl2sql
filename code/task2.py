import re
import random
import cn2an
import math
import json
import thulac
import argparse
from collections import defaultdict
from zhon import hanzi

import keras.backend as K
from nl2sql.utils import read_data, read_tables
from keras_bert import get_checkpoint_paths, load_vocabulary, Tokenizer, load_trained_model_from_checkpoint
from keras.utils.data_utils import Sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback


thu1 = thulac.thulac()

CN_UNIT = {
    '十' : 10,
    '拾' : 10,
    '百' : 100,
    '佰' : 100,
    '千' : 1000,
    '仟' : 1000,
    '万' : 10000,
    '萬' : 10000,
    '亿' : 100000000,
    '億' : 100000000,
    '兆' : 1000000000000,
}

cn_num = '〇一二三四五六七八九零壹贰叁肆伍陆柒捌玖貮两'

def isfloat(value):
    try:
        v = float(value)
        return True
    except ValueError:
        return False


def cn_to_an(string):
    try:
        return str(cn2an.cn2an(string, 'normal'))
    except:
        return string


def an_to_cn(string):
    try:
        return str(cn2an.an2cn(string))
    except:
        return string


def convert_num(string):
    string = string.replace('百分之', '')
    string = string.replace('第', '')
    string = string.replace('个', '')
    result = [cn_to_an(string)]

    max_value = 0
    for k in CN_UNIT:
        pos_idx = string.find(k)
        if pos_idx != -1:
            max_value = cn_to_an(string[: pos_idx])
    if max_value != 0 and max_value != '':
        result.append(max_value)
    return result


def convert_year(string):
    year = string.replace('年', '')
    year = cn_to_an(year)
    if isfloat(year) and float(year) < 1900:
        year = int(year) + 2000
        return str(year)
    else:
        return string


def extract_value_in_question(question):
    all_values = []
    num_values = re.findall(r'[-+]?[0-9]*\.?[0-9]+', question)
    all_values += num_values

    num_year_values = re.findall(r'[0-9][0-9]年', question)
    all_values += ['20{}'.format(v[:-1]) for v in num_year_values]

    cn_year_values = re.findall(r'[{}][{}]年'.format(cn_num, cn_num), question)
    all_values += [convert_year(v) for v in cn_year_values]

    if '负' in question:
        all_values.append('0')

    q_cut = thu1.cut(question)
    cn_num_values = []
    for word, tag in q_cut:
        if tag == 'm' and word not in ['一下', '几']:
            if word in all_values:
                continue
            num = re.findall(r'[-+]?[0-9]*\.?[0-9]+', word)
            num_cn_map = {n: an_to_cn(n) for n in num}
            for n in num:
                word = word.replace(n, num_cn_map[n])
            cn_num_values += convert_num(word)
        if tag == 't':
            year_value = convert_year(word)
            if year_value != word:
                cn_num_values.append(year_value)
    all_values += cn_num_values
    return list(set(all_values))


def generate_more_conds_nl(header, value_list):
    conds_idx = 0
    pattern = "{}大于{}"
    return {(conds_idx, v): pattern.format(header, v) for v in value_list}


def generate_less_conds_nl(header, value_list):
    conds_idx = 1
    pattern = "{}小于{}"
    return {(conds_idx, v): pattern.format(header, v) for v in value_list}


def generate_equal_conds_nl(header, value_list):
    conds_idx = 2
    pattern = "{}是{}"
    return {(conds_idx, v): pattern.format(header, v) for v in value_list}


def generate_nonequal_conds_nl(header, value_list):
    conds_idx = 3
    pattern = "{}不是{}"
    return {(conds_idx, v): pattern.format(header, v) for v in value_list}


def generate_conds_value(data, header):
    q = data.question.text
    col_v = [v for v in data.table.df[header] if len(v) < 20 and len(set(q) & set(v)) > 0]
    return list(set(col_v))


def generate_real_conds_nl(header, value_list):
    result = {}
    result.update(generate_more_conds_nl(header, value_list))
    result.update(generate_less_conds_nl(header, value_list))
    result.update(generate_equal_conds_nl(header, value_list))
    #result.update(generate_nonequal_conds_nl(header, value_list))
    return result


def generate_text_conds_nl(header, value_list):
    result = {}
    result.update(generate_equal_conds_nl(header, value_list))
    #result.update(generate_nonequal_conds_nl(header, value_list))
    return result


def synthesis_conds_nl(data):
    nl_text = {}
    nl_real = {}
    conds = {}
    if data.sql:
        conds = {tuple(c):1 for c in data.sql.conds}
    value_in_question = extract_value_in_question(data.question.text)

    for idx, header in enumerate(data.table.header):
        h = header[0]
        value_in_table = generate_conds_value(data, h)

        if header[1] == 'text':
            syn_nl = generate_text_conds_nl(h, value_in_table)
            syn_nl = {(idx, k[0], k[1]):v for k, v in syn_nl.items()}
            nl_text.update(syn_nl)
        elif header[1] == 'real':
            if len(value_in_table) == 1:
                syn_nl = generate_real_conds_nl(h, value_in_question + value_in_table)
            else:
                syn_nl = generate_real_conds_nl(h, value_in_question)
            syn_nl = {(idx, k[0], k[1]):v for k, v in syn_nl.items()}
            nl_real.update(syn_nl)

    all_nl = {}
    all_nl.update(nl_real)
    all_nl.update(nl_text)

    nl_with_label = [(nl, 1) if k in conds else (nl, 0)
                     for k, nl in all_nl.items()]
    conds_nl_to_sql = {v.lower(): k for k, v in all_nl.items()}
    return nl_with_label, conds_nl_to_sql


def synthesis_nl_pair(train_data):
    all_pair = []
    nl_to_sql = {}
    for data in train_data:
        nl_with_label, conds_nl_to_sql = synthesis_conds_nl(data)
        all_pair += [(data.question.text.lower(), syn_nl, label)
                     for syn_nl, label in nl_with_label]
        nl_to_sql[data.question.text.lower()] = conds_nl_to_sql
    return all_pair, nl_to_sql


class DataSequence(Sequence):
    def __init__(self, data, tokenizer, is_train=True, max_len=120,
                 batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.is_train = is_train
        self.max_len = max_len
        self._global_indices = np.arange(len(data))
        self.on_epoch_end()

    def _pad_sequences(self, seqs, max_len=None):
        return pad_sequences(seqs, maxlen=max_len, padding='post')

    def __getitem__(self, batch_id):
        batch_data_indices = \
            self._global_indices[batch_id * self.batch_size: (batch_id + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_data_indices]

        X1, X2 = [], []
        Y = []

        for question, syn_nl, label in batch_data:
            x1, x2 = self.tokenizer.encode(first=question, second=syn_nl)
            X1.append(x1)
            X2.append(x2)
            if self.is_train:
                Y.append([label])

        X1 = self._pad_sequences(X1, max_len=self.max_len)
        X2 = self._pad_sequences(X2, max_len=self.max_len)
        inputs = {'input_x1': X1, 'input_x2': X2}
        if self.is_train:
            Y = self._pad_sequences(Y, max_len=1)
            outputs = {'output_similarity': Y}
            return inputs, outputs
        else:
            return inputs

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._global_indices)


class SimpleTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R


def construct_model(paths):
    token_dict = load_vocabulary(paths.vocab)
    tokenizer = SimpleTokenizer(token_dict)

    bert_model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=None)
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,), name='input_x1', dtype='int32')
    x2_in = Input(shape=(None,), name='input_x2')
    x = bert_model([x1_in, x2_in])
    x_cls = Lambda(lambda x: x[:, 0])(x)
    y_pred = Dense(1, activation='sigmoid', name='output_similarity')(x_cls)

    model = Model([x1_in, x2_in], y_pred)
    model.compile(loss={'output_similarity':'binary_crossentropy'},
                  optimizer=Adam(1e-5),
                  metrics={'output_similarity': 'accuracy'})

    return model, tokenizer


def merge_question_values(nl_pair, nl_map, model_result):
    result = defaultdict(list)
    for question, pred in zip(nl_pair, model_result):
        sel_sql = nl_map[question[0]][question[1].lower()]
        result[question[0]].append((sel_sql, pred[0]))
    return result


def deduplicate_conds(select_conds):
    select_col_op = {}
    output_result = []

    sort_result = sorted(select_conds, key=lambda x: x[1], reverse=True)
    for select_col, p in sort_result:
        op = select_col[1]
        col = select_col[0]
        if op < 2 and (col, op) in select_col_op:
            continue
        output_result.append(select_col)
        select_col_op[(col, op)] = 1
    return output_result


def find_match_values(conds_pred, current_output, sorted_result):
    select_col_op = [v[:2] for v in current_output]
    for v, p in sorted_result: # find value with the same column and op
        if v[:2] in conds_pred and v[:2] not in select_col_op:
            current_output.append(v)
            select_col_op.append(v[:2])

    if len(conds_pred) > len(current_output): # find value with same column
        select_col = [v[0] for v in current_output]
        conds_col = {v[0]: v for v in conds_pred}
        for v, p in sorted_result:
            if v[0] in conds_col and v[0] not in select_col:
                col_op = conds_col[v[0]]
                current_output.append((col_op[0], col_op[1], v[2]))
                select_col.append(v[0])


def select_values(data, task1_pred, task2_pred, t):
    output_result = [r for r in task2_pred if r[1] > t]
    output_result = deduplicate_conds(output_result)
    sorted_result = sorted(task2_pred, key=lambda x: x[1], reverse=True)

    for idx, header in enumerate(data.table.header):
        if header[1] == 'text':
            col_v = [v for v in data.table.df[header[0]] if len(v) >= 20]
            for v in col_v:
                format_v = re.sub("[{}]+$".format(hanzi.punctuation), '', v)
                if format_v in data.question.text:
                    output_result = [(idx, 2, v)] + output_result

    if task1_pred['cond_conn_op'] == 0:
        output_result = output_result[:1]
    elif task1_pred['cond_conn_op'] != 0 and len(output_result) < 2:
        conds_pred = [tuple(c) for c in task1_pred['conds']]
        find_match_values(conds_pred, output_result, sorted_result[:10])

    if not output_result and sorted_result:
        output_result.append(sorted_result[0][0])
    return list(set(output_result))


def load_preds(pred_file):
    result = []
    with open(pred_file) as file:
        for line in file:
            result.append(json.loads(line))
    return result


def train(opt):
    train_tables = read_tables(opt.train_table_file)
    train_data = read_data(opt.train_data_file, train_tables)
    train_pair, train_map = synthesis_nl_pair(train_data[:])

    random.seed(666)
    positive_pair = [p for p in train_pair if p[2] == 1]
    negative_pair = [p for p in train_pair if p[2] == 0]
    negative_sample = random.sample(negative_pair, len(positive_pair) * 10)
    train_sample = positive_pair + negative_sample

    paths = get_checkpoint_paths(opt.bert_model)
    model, tokenizer = construct_model(paths)
    train_iter = DataSequence(train_sample, tokenizer, batch_size=48, max_len=120)
    model.fit_generator(train_iter, epochs=5, workers=4)
    model.save_weights('../model/task2.h5')


def predict(opt):
    test_tables = read_tables(opt.test_table_file)
    test_data = read_data(opt.test_data_file, test_tables)[:]
    test_pair, test_map = synthesis_nl_pair(test_data)

    paths = get_checkpoint_paths(opt.bert_model)
    model, tokenizer = construct_model(paths)
    model.load_weights('../model/task2.h5')

    test_iter = DataSequence(test_pair, tokenizer, batch_size=48, shuffle=False)
    test_preds = model.predict_generator(test_iter, verbose=1)

    task2_preds = merge_question_values(test_pair, test_map, test_preds)
    task1_preds = load_preds(opt.task1_output)

    for data, t1_preds in zip(test_data, task1_preds):
        t2_preds = task2_preds[data.question.text.lower()]
        select_value = select_values(data, t1_preds, t2_preds, 0.995)
        t1_preds['conds'] = [list(v) for v in select_value]

    with open(opt.task1_output, 'w') as f:
        for item in task1_preds:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--train_data_file',
                              default='../data/train/train.json')
    train_parser.add_argument('--train_table_file',
                              default='../data/train/train.tables.json')
    train_parser.add_argument('--bert_model',
                              default='../model/chinese_wwm_L-12_H-768_A-12')
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser('infer')
    infer_parser.add_argument('--test_data_file',
                              default='../data/test/test.json')
    infer_parser.add_argument('--test_table_file',
                              default='../data/test/test.tables.json')
    infer_parser.add_argument('--bert_model',
                              default='../model/chinese_wwm_L-12_H-768_A-12')
    infer_parser.add_argument('--task1_output',
                              required=True, default='../submit/task1_output.json')
    infer_parser.set_defaults(func=infer)

    opt = parser.parse_args()
    opt.func(opt)


if __name__ == "__main__":
    main()
