from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
import pickle
import pandas as pd
import torch
import numpy as np
from typing import List, Text, Optional, Tuple
import gc
import math
from configs import get_config, AttributeDict

import utils
from generator.vocab_mof import MOFVocab
from generator.vocab_y import PropVocab
from hgraph.mol_graph import MolGraph

torch.multiprocessing.set_sharing_strategy('file_system')

DataTuple = Tuple[Text, Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]


def graphs_to_tuples(df: pd.DataFrame, smiles_column: Text, vocab_mof, vocab_y) -> List[DataTuple]:
    n = len(df)
    smiles_df = df[[smiles_column]]
    smiles_df = smiles_df.values

    mof_ids = vocab_mof.df_to_ids(df)

    has_y = all(i in df.columns.tolist() for i in vocab_y.labels)

    y, y_mask = vocab_y.df_to_y(df) if has_y else utils.invalid_values(len(vocab_y.labels), n)
    return list(zip(*[smiles_df, mof_ids, y, y_mask]))


def to_numpy(tensors, outs):
    convert = lambda x: x.numpy() if type(x) is torch.Tensor else x
    a, b, c = tensors
    d = outs
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c, d


def tensorize(mol_batch, vocab_x, vocab_atom):
    outs = {'x': np.vstack([t[0][0] for t in mol_batch]), 'mof': None, 'y': None, 'y_mask': None}
    mof_arr = np.vstack([t[1] for t in mol_batch]).astype(int)
    outs['mof'] = torch.from_numpy(mof_arr)
    y_arr = np.vstack([t[2] for t in mol_batch]).astype(np.float32)
    outs['y'] = torch.from_numpy(y_arr)
    y_mask = np.vstack([t[3] for t in mol_batch]).astype(np.float32)
    outs['y_mask'] = torch.from_numpy(y_mask)
    mol_batch_smiles = [x[0][0] for x in mol_batch]
    x = MolGraph.tensorize(mol_batch_smiles, vocab_x, vocab_atom)
    return to_numpy(x, outs)


def df_cleaning(config: AttributeDict):
    df = pd.read_csv(config['df_csv_file'], na_values=['', 'NA', 'NaN'], keep_default_na=False)
    df_prop = pd.read_csv(config['df_prop_csv_file'], na_values=['', 'NA', 'NaN'], keep_default_na=False)

    if config['col_x'] not in df.columns:
        df = df.rename(columns={'SMILES': config['col_x']})

    index = df['id2mof'].drop_duplicates().index  # 去重之后的索引
    df_drop_same = df.loc[index].reset_index()  # 根据索引对数据进行去重
    index = df_drop_same['id2mof'].drop_duplicates().index
    sub_df = df_drop_same.loc[index].sort_values(by='id2mof')
    mof2ids = OrderedDict()
    ids2mof = OrderedDict()
    for _, row in sub_df.iterrows():
        mof = (row['metal_node'], row['organic_core'], row['topology'])
        ids2mof[row['id2mof']] = mof
        mof2ids[mof] = row['id2mof']

    def valid_mof(x):
        return tuple(x) in mof2ids.keys()

    df_prop = df_prop.query('mask').reset_index(drop=True)
    valid_mofs_list = df_prop[config['col_mof']].apply(valid_mof, axis=1).tolist()
    df_prop = df_prop[valid_mofs_list].reset_index(drop=True)
    df_prop['id2mof'] = df_prop[config['col_mof']].apply(lambda x: mof2ids[tuple(x)], axis=1)

    train_index = np.array(df[df[config['col_testtrain']] == 1].index.tolist())
    test_index = np.array(df[df[config['col_testtrain']] == 0].index.tolist())

    prop_train_index = np.array(df_prop[df_prop[config['col_testtrain']] == 1].index.tolist())
    prop_test_index = np.array(df_prop[df_prop[config['col_testtrain']] == 0].index.tolist())

    df_train = df.loc[train_index]
    df_test = df.loc[test_index]
    df_prop_train = df_prop.loc[prop_train_index]
    df_prop_test = df_prop.loc[prop_test_index]

    with open(config['file_ids2mof'], 'wb') as file_ids2mof:
        pickle.dump(ids2mof, file_ids2mof)
    with open(config['file_mof2ids'], 'wb') as file_mof2ids:
        pickle.dump(mof2ids, file_mof2ids)
    with open(config['file_df_train'], 'wb') as file_df_train:
        pickle.dump(df_train, file_df_train)
    with open(config['file_df_test'], 'wb') as file_df_test:
        pickle.dump(df_test, file_df_test)
    with open(config['file_df_prop_train'], 'wb') as file_df_prop_train:
        pickle.dump(df_prop_train, file_df_prop_train)
    with open(config['file_df_prop_test'], 'wb') as file_df_prop_test:
        pickle.dump(df_prop_test, file_df_prop_test)
    return df_train, df_test, df_prop_train, df_prop_test


def process(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        mol = MolGraph(s)
        for node, attr in mol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add(attr['label'])
            for i, s in attr['inter_label']:
                vocab.add((smiles, s))
    return vocab


def get_vocabs(config: AttributeDict, df_train, df_prop_train):
    df_smiles = pd.concat([df_train[config['col_x']], df_prop_train[config['col_x']]]).unique()
    df_smiles = list(set(df_smiles))

    vocab_batch_size = len(df_smiles) // config['cpu_cores_num'] + 1
    batches = [df_smiles[i: i + vocab_batch_size] for i in range(0, len(df_smiles), vocab_batch_size)]

    pool = Pool(config['cpu_cores_num'])
    vocab_list = pool.map(process, batches)

    vocab = [(x, y) for vocab in vocab_list for x, y in vocab]
    vocab = list(set(vocab))

    with open(config['file_vocab_x_txt'], "w", encoding="utf-8") as vocab_x_txt:
        for x, y in sorted(vocab):
            vocab_x_txt.write(f"{x} {y}\n")

    vocab_mof = MOFVocab.from_data(pd.concat([df_train, df_prop_train], axis=0, ignore_index=False, sort=False),
                                   config['col_mof'])
    vocab_y = PropVocab.from_data(df_prop_train, config['col_y'],
                                  config['y_weights'], config['y_scaler_type'])

    with open(config['file_vocab_mof'], 'wb') as file_vocab_mof:
        pickle.dump(vocab_mof, file_vocab_mof)

    with open(config['file_vocab_y'], 'wb') as file_vocab_y:
        pickle.dump(vocab_y, file_vocab_y)


def get_tensors(config: AttributeDict, df_train, df_test, df_prop_train, df_prop_test,
                vocab_mof, vocab_y, vocab_x, vocab_atom):
    train_tuple_mof = graphs_to_tuples(df_train, config['col_x'], vocab_mof, vocab_y)
    test_tuple_mof = graphs_to_tuples(df_test, config['col_x'], vocab_mof, vocab_y)
    prop_train_tuple_mof = graphs_to_tuples(df_prop_train, config['col_x'], vocab_mof, vocab_y)
    prop_test_tuple_mof = graphs_to_tuples(df_prop_test, config['col_x'], vocab_mof, vocab_y)

    df_train_tuple = prop_train_tuple_mof + train_tuple_mof
    df_test_tuple = prop_test_tuple_mof + test_tuple_mof

    func = partial(tensorize, vocab_x=vocab_x, vocab_atom=vocab_atom)

    tensors_list = [df_train_tuple, df_test_tuple, prop_train_tuple_mof, prop_test_tuple_mof]
    tensors_file_name = [config['tensors_train'], config['tensors_test'], config['tensors_train_prop'],
                         config['tensors_test_prop']]

    for num_list, tensors in enumerate(tensors_list):
        total_chunks = math.ceil(len(tensors) / 50000)
        step = 0
        for num_chunks in range(total_chunks):
            tensors_chunks = tensors[num_chunks * 50000: (num_chunks * 50000 + 50000)]
            batches = [tensors_chunks[i: i + config['train_batch_size']] for i in
                       range(0, len(tensors_chunks), config['train_batch_size'])]
            with Pool(processes=config['cpu_cores_num']) as pool:
                all_data = pool.map(func, batches)

            pool.close()
            pool.join()
            del batches
            gc.collect()

            num_splits = max(len(all_data) // 1000, 1)
            le = (len(all_data) + num_splits - 1) // num_splits
            for split_id in range(num_splits):
                st = split_id * le
                sub_data = all_data[st: st + le]
                with open(tensors_file_name[num_list] + '/tensors-%d.pkl' % step, 'wb') as f:
                    pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
                step += 1


if __name__ == "__main__":
    config: AttributeDict = get_config()
    utils.set_seed(config['generator_rand_seed'])

    # files in "data/df/"
    config['df_cleaning'] = False
    # files in "results/vocabs/"
    # for vocab_x, you may need to add some additional motif vocabulary.
    config['get_vocabs'] = False
    # files in "data/tensors/"
    config['get_df_tensors'] = True

    if config['df_cleaning']:
        df_train, df_test, df_prop_train, df_prop_test = df_cleaning(config)
    else:
        with open(config['file_df_train'], 'rb') as file_df_train:
            df_train = pickle.load(file_df_train)
        with open(config['file_df_test'], 'rb') as file_df_test:
            df_test = pickle.load(file_df_test)
        with open(config['file_df_prop_train'], 'rb') as file_df_prop_train:
            df_prop_train = pickle.load(file_df_prop_train)
        with open(config['file_df_prop_test'], 'rb') as file_df_prop_test:
            df_prop_test = pickle.load(file_df_prop_test)

    if config['get_vocabs']:
        get_vocabs(config, df_train, df_prop_train)

    vocab_mof, vocab_y, vocab_x, vocab_atom = utils.set_vocab(config)

    if config['get_df_tensors']:
        get_tensors(config, df_train, df_test, df_prop_train, df_prop_test, vocab_mof, vocab_y, vocab_x, vocab_atom)
