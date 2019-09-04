import argparse
import json
import math
import os
import re

import keras.backend as K
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Dense, Input, Lambda, Masking, Multiply, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import Sequence
from keras_bert import (Tokenizer, get_checkpoint_paths,
                        build_model_from_config,
                        load_trained_model_from_checkpoint, load_vocabulary)
from tqdm import tqdm

from nl2sql.utils import (SQL, MultiSentenceTokenizer, Query, read_data,
                          read_tables)
from nl2sql.utils.loss import custom_sparse_categorical_crossentropy
from nl2sql.utils.optimizer import RAdam


def remove_brackets(s):
    return re.sub(r'[\(\（].*[\)\）]', '', s)


class QueryTokenizer(MultiSentenceTokenizer):
    col_type_token_dict = {'text': '[unused11]', 'real': '[unused12]'}

    def tokenize(self, query: Query, col_orders=None):
        question_tokens = [self._token_cls] + \
            self._tokenize(query.question.text)
        header_tokens = []

        if col_orders is None:
            col_orders = np.arange(len(query.table.header))

        header = [query.table.header[i] for i in col_orders]

        for col_name, col_type in header:
            col_type_token = self.col_type_token_dict[col_type]
            col_name = remove_brackets(col_name)
            col_name_tokens = self._tokenize(col_name)
            header_tokens.append([col_type_token] + col_name_tokens)

        all_tokens = [question_tokens] + header_tokens
        return self._pack(*all_tokens)

    def encode(self, query: Query, col_orders=None):
        tokens, tokens_lens = self.tokenize(query, col_orders)
        token_ids = self._convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        header_indices = np.cumsum(tokens_lens)
        return token_ids, segment_ids, header_indices[:-1]


class SqlLabelEncoder:

    def encode(self, sql: SQL, num_cols):
        cond_conn_op_label = sql.cond_conn_op

        sel_agg_label = np.ones(
            num_cols, dtype='int32') * len(SQL.agg_sql_dict)
        for col_id, agg_op in zip(sql.sel, sql.agg):
            if col_id < num_cols:
                sel_agg_label[col_id] = agg_op

        cond_op_label = np.ones(num_cols, dtype='int32') * len(SQL.op_sql_dict)
        for col_id, cond_op, _ in sql.conds:
            if col_id < num_cols:
                cond_op_label[col_id] = cond_op

        return cond_conn_op_label, sel_agg_label, cond_op_label

    def decode(self, cond_conn_op_label, sel_agg_label, cond_op_label):
        cond_conn_op = int(cond_conn_op_label)
        sel, agg, conds = [], [], []

        for col_id, (agg_op, cond_op) in enumerate(zip(sel_agg_label, cond_op_label)):
            if agg_op < len(SQL.agg_sql_dict):
                sel.append(col_id)
                agg.append(int(agg_op))
            if cond_op < len(SQL.op_sql_dict):
                conds.append([col_id, int(cond_op)])
        return {
            'sel': sel,
            'agg': agg,
            'cond_conn_op': cond_conn_op,
            'conds': conds
        }


class DataSequence(Sequence):
    def __init__(self, data, tokenizer, label_encoder, is_train=True, max_len=160, batch_size=32, shuffle=True, shuffle_header=True, global_indices=None):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.shuffle = shuffle
        self.shuffle_header = shuffle_header
        self.is_train = is_train
        self.max_len = max_len
        if global_indices is None:
            self._global_indices = np.arange(len(data))
        else:
            self._global_indices = global_indices
        if shuffle:
            np.random.shuffle(self._global_indices)

    def _pad_sequences(self, seqs, max_len=None):
        padded = pad_sequences(seqs, maxlen=None, padding='post')
        if max_len is not None:
            padded = padded[:, :max_len]
        return padded

    def __getitem__(self, batch_id):
        batch_data_indices = \
            self._global_indices[batch_id *
                                 self.batch_size: (batch_id + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_data_indices]

        TOKEN_IDS, SEGMENT_IDS = [], []
        HEADER_IDS, HEADER_MASK = [], [],

        COND_CONN_OP = []
        SEL_AGG = []
        COND_OP = []

        for query in batch_data:
            table = query.table

            col_orders = np.arange(len(table.header))
            if self.shuffle_header:
                np.random.shuffle(col_orders)

            token_ids, segment_ids, header_ids = self.tokenizer.encode(
                query, col_orders)
            header_ids = [hid for hid in header_ids if hid < self.max_len]
            header_mask = [1] * len(header_ids)
            col_orders = col_orders[: len(header_ids)]

            TOKEN_IDS.append(token_ids)
            SEGMENT_IDS.append(segment_ids)
            HEADER_IDS.append(header_ids)
            HEADER_MASK.append(header_mask)

            if not self.is_train:
                continue
            sql = query.sql

            cond_conn_op, sel_agg, cond_op = self.label_encoder.encode(
                sql, num_cols=len(table.header))

            sel_agg = sel_agg[col_orders]
            cond_op = cond_op[col_orders]

            COND_CONN_OP.append(cond_conn_op)
            SEL_AGG.append(sel_agg)
            COND_OP.append(cond_op)

        TOKEN_IDS = self._pad_sequences(TOKEN_IDS, max_len=self.max_len)
        SEGMENT_IDS = self._pad_sequences(SEGMENT_IDS, max_len=self.max_len)
        HEADER_IDS = self._pad_sequences(HEADER_IDS)
        HEADER_MASK = self._pad_sequences(HEADER_MASK)

        inputs = {
            'input_token_ids': TOKEN_IDS,
            'input_segment_ids': SEGMENT_IDS,
            'input_header_ids': HEADER_IDS,
            'input_header_mask': HEADER_MASK
        }

        if self.is_train:
            SEL_AGG = self._pad_sequences(SEL_AGG)
            SEL_AGG = np.expand_dims(SEL_AGG, axis=-1)
            COND_CONN_OP = np.expand_dims(COND_CONN_OP, axis=-1)
            COND_OP = self._pad_sequences(COND_OP)
            COND_OP = np.expand_dims(COND_OP, axis=-1)

            outputs = {
                'output_sel_agg': SEL_AGG,
                'output_cond_conn_op': COND_CONN_OP,
                'output_cond_op': COND_OP
            }
            return inputs, outputs
        else:
            return inputs

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._global_indices)


def build_model(bert_model_path, load_bert_weights=False):
    def seq_gather(x):
        seq, idxs = x
        idxs = K.cast(idxs, 'int32')
        return K.tf.batch_gather(seq, idxs)

    paths = get_checkpoint_paths(bert_model_path)

    if load_bert_weights == True:
        bert_model = load_trained_model_from_checkpoint(
            paths.config, paths.checkpoint, seq_len=None)
    else:
        bert_model, _ = build_model_from_config(
            paths.config,
            seq_len=None)

    for l in bert_model.layers:
        l.trainable = True
    num_sel_agg = len(SQL.agg_sql_dict) + 1
    num_cond_op = len(SQL.op_sql_dict) + 1
    num_cond_conn_op = len(SQL.conn_sql_dict)

    inp_token_ids = Input(shape=(None,), name='input_token_ids', dtype='int32')
    inp_segment_ids = Input(
        shape=(None,), name='input_segment_ids', dtype='int32')
    inp_header_ids = Input(
        shape=(None,), name='input_header_ids', dtype='int32')
    inp_header_mask = Input(shape=(None, ), name='input_header_mask')

    x = bert_model([inp_token_ids, inp_segment_ids])  # (None, seq_len, 768)

    # predict cond_conn_op
    x_for_cond_conn_op = Lambda(lambda x: x[:, 0])(x)  # (None, 768)
    p_cond_conn_op = Dense(num_cond_conn_op, activation='softmax',
                           name='output_cond_conn_op')(x_for_cond_conn_op)

    # predict sel_agg
    x_for_header = Lambda(seq_gather, name='header_seq_gather')(
        [x, inp_header_ids])  # (None, h_len, 768)
    header_mask = Lambda(lambda x: K.expand_dims(
        x, axis=-1))(inp_header_mask)  # (None, h_len, 1)

    x_for_header = Multiply()([x_for_header, header_mask])
    x_for_header = Masking()(x_for_header)

    p_sel_agg = Dense(num_sel_agg, activation='softmax',
                      name='output_sel_agg')(x_for_header)

    x_for_cond_op = Concatenate(axis=-1)([x_for_header, p_sel_agg])
    p_cond_op = Dense(num_cond_op, activation='softmax',
                      name='output_cond_op')(x_for_cond_op)

    model = Model(
        [inp_token_ids, inp_segment_ids, inp_header_ids, inp_header_mask],
        [p_cond_conn_op, p_sel_agg, p_cond_op]
    )
    return model


class LearningRateScheduler(Callback):
    def __init__(self, init_lr, min_lr):
        self.passed = 0
        self.init_lr = init_lr
        self.min_lr = min_lr

    def on_batch_begin(self, batch, logs=None):
        if self.passed < self.params['steps']:
            lr = (self.passed + 1) / self.params['steps'] * self.init_lr
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1) /
                  self.params['steps']) * (self.init_lr - self.min_lr)
            lr += self.min_lr
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1


class EvaluateCallback(Callback):
    def __init__(self, val_dataseq):
        self.val_dataseq = val_dataseq

    def on_epoch_end(self, epoch, logs=None):
        val_dataseq = self.val_dataseq
        is_train = self.val_dataseq.is_train
        self.val_dataseq.is_train = False
        pred_sqls = []
        for batch_data in self.val_dataseq:
            header_lens = np.sum(batch_data['input_header_mask'], axis=-1)
            preds_cond_conn_op, preds_sel_agg, preds_cond_op = self.model.predict_on_batch(
                batch_data)
            sqls = outputs_to_sqls(preds_cond_conn_op, preds_sel_agg, preds_cond_op,
                                   header_lens, val_dataseq.label_encoder)
            pred_sqls += sqls
        conn_correct = 0
        agg_correct = 0
        conds_correct = 0
        conds_col_id_correct = 0
        all_correct = 0
        num_queries = len(self.val_dataseq.data)

        true_sqls = [query.sql for query in self.val_dataseq.data]
        for pred_sql, true_sql in zip(pred_sqls, true_sqls):
            n_correct = 0
            if pred_sql['cond_conn_op'] == true_sql.cond_conn_op:
                conn_correct += 1
                n_correct += 1

            pred_aggs = set(zip(pred_sql['sel'], pred_sql['agg']))
            true_aggs = set(zip(true_sql.sel, true_sql.agg))
            if pred_aggs == true_aggs:
                agg_correct += 1
                n_correct += 1

            pred_conds = set([(cond[0], cond[1])
                              for cond in pred_sql['conds']])
            true_conds = set([(cond[0], cond[1]) for cond in true_sql.conds])

            if pred_conds == true_conds:
                conds_correct += 1
                n_correct += 1

            pred_conds_col_ids = set([cond[0] for cond in pred_sql['conds']])
            true_conds_col_ids = set([cond[0] for cond in true_sql['conds']])
            if pred_conds_col_ids == true_conds_col_ids:
                conds_col_id_correct += 1

            if n_correct == 3:
                all_correct += 1

        print('conn_acc: {}'.format(conn_correct / num_queries))
        print('agg_acc: {}'.format(agg_correct / num_queries))
        print('conds_acc: {}'.format(conds_correct / num_queries))
        print('conds_col_id_acc: {}'.format(
            conds_col_id_correct / num_queries))
        print('total_acc: {}'.format(all_correct / num_queries))
        logs['val_tot_acc'] = all_correct / num_queries
        self.val_dataseq.is_train = is_train


def outputs_to_sqls(preds_cond_conn_op, preds_sel_agg, preds_cond_op, header_lens, label_encoder):
    preds_cond_conn_op = np.argmax(preds_cond_conn_op, axis=-1)
    # preds_sel_agg = np.argmax(preds_sel_agg, axis=-1)
    preds_cond_op = np.argmax(preds_cond_op, axis=-1)

    sqls = []

    for cond_conn_op, sel_agg, cond_op, header_len in zip(preds_cond_conn_op,
                                                          preds_sel_agg,
                                                          preds_cond_op,
                                                          header_lens):
        sel_agg = sel_agg[:header_len]
        sel_agg[sel_agg == sel_agg[:, :-1].max()] = 1
        sel_agg = np.argmax(sel_agg, axis=-1)

        sql = label_encoder.decode(cond_conn_op, sel_agg, cond_op)
        sql['conds'] = [cond for cond in sql['conds'] if cond[0] < header_len]
        sel = []
        agg = []
        for col_id, agg_type in zip(sql['sel'], sql['agg']):
            if col_id < header_len:
                sel.append(col_id)
                agg.append(agg_type)
        sql['sel'] = sel
        sql['agg'] = agg
        sqls.append(sql)
    return sqls


def train(opt):
    train_tables = read_tables(opt.train_table_file)
    train_data = read_data(opt.train_data_file, train_tables)
    val_tables = read_tables(opt.val_table_file)
    val_data = read_data(opt.val_data_file, val_tables)

    paths = get_checkpoint_paths(opt.bert_model)
    token_dict = load_vocabulary(paths.vocab)
    query_tokenizer = QueryTokenizer(token_dict)
    label_encoder = SqlLabelEncoder()

    train_dataseq = DataSequence(
        data=train_data,
        tokenizer=query_tokenizer,
        label_encoder=label_encoder,
        is_train=True,
        max_len=160,
        batch_size=opt.batch_size
    )

    val_dataseq = DataSequence(
        data=val_data,
        tokenizer=query_tokenizer,
        label_encoder=label_encoder,
        is_train=True,
        max_len=160,
        shuffle=False,
        shuffle_header=False,
        batch_size=opt.batch_size
    )

    model = build_model(opt.bert_model)
    model.compile(
        loss=custom_sparse_categorical_crossentropy,
        optimizer=Adam(5e-5)
    )
    callbacks = [
        LearningRateScheduler(5e-5, min_lr=1e-5),
        EvaluateCallback(val_dataseq),
        ModelCheckpoint(
            os.path.join(
                opt.model_dir,
                "task1.{epoch:02d}-{val_tot_acc:.3f}.hd5"
            ),
            monitor='val_tot_acc',
            mode='max',
            save_best_only=True
        )
    ]
    model.fit_generator(train_dataseq, epochs=1, callbacks=callbacks)


def infer(opt):
    test_tables = read_tables(opt.test_table_file)
    test_data = read_data(opt.test_data_file, test_tables)
    print('read {} test queries'.format(len(test_data)))
    paths = get_checkpoint_paths(opt.bert_model)
    token_dict = load_vocabulary(paths.vocab)
    query_tokenizer = QueryTokenizer(token_dict)
    model = build_model(opt.bert_model)
    model.load_weights(opt.model_weights)
    query_tokenizer = QueryTokenizer(token_dict)
    label_encoder = SqlLabelEncoder()
    test_dataseq = DataSequence(
        data=test_data,
        tokenizer=query_tokenizer,
        label_encoder=label_encoder,
        is_train=False,
        max_len=160,
        shuffle=False,
        shuffle_header=False,
        batch_size=opt.batch_size
    )
    pred_sqls = []

    for batch_data in tqdm(test_dataseq):
        header_lens = np.sum(batch_data['input_header_mask'], axis=-1)
        preds_cond_conn_op, preds_sel_agg, preds_cond_op = \
            model.predict_on_batch(batch_data)
        sqls = outputs_to_sqls(preds_cond_conn_op, preds_sel_agg, preds_cond_op,
                               header_lens, test_dataseq.label_encoder)
        pred_sqls += sqls

    with open(opt.output_file, 'w') as f:
        for sql in pred_sqls:
            f.write(json.dumps(sql, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--batch_size', type=int, default=64)
    train_parser.add_argument('--model_dir', required=True)
    train_parser.add_argument(
        '--bert_model',
        default='../model/chinese_wwm_L-12_H-768_A-12'
    )
    train_parser.add_argument(
        '--train_table_file',
        default='../data/train/train.tables.json')
    train_parser.add_argument(
        '--train_data_file',
        default='../data/train/train.json'
    )
    train_parser.add_argument(
        '--val_table_file',
        default='../data/val/val.tables.json')
    train_parser.add_argument(
        '--val_data_file',
        default='../data/val/val.json'
    )
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser('infer')
    infer_parser.add_argument(
        '--bert_model',
        default='../model/chinese_wwm_L-12_H-768_A-12'
    )
    infer_parser.add_argument('--model_weights', required=True)
    infer_parser.add_argument('--batch_size', type=int, default=64)
    infer_parser.add_argument(
        '--test_table_file',
        default='../data/test/test.tables.json'
    )
    infer_parser.add_argument(
        '--test_data_file',
        default='../data/test/test.json'
    )
    infer_parser.add_argument(
        '--output_file',
        required=True,
        default='../submit/task1_output.json'
    )

    infer_parser.set_defaults(func=infer)

    opt = parser.parse_args()
    opt.func(opt)


if __name__ == "__main__":
    main()
