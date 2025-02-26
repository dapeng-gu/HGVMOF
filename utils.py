import pickle
import random
import re
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sklearn.metrics
import pandas as pd
from torch.nn import functional as F

from configs import AttributeDict

from generator.vocab_atom import VocabAtom
from generator.vocab_x import PairVocab
from optimizer.scscore import SCScorer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_vocab(config: AttributeDict):
    with open(config['file_vocab_mof'], 'rb') as file_vocab_mof:
        vocab_mof = pickle.load(file_vocab_mof)
    with open(config['file_vocab_y'], 'rb') as file_vocab_y:
        vocab_y = pickle.load(file_vocab_y)

    vocab_x = [x.strip("\r\n ").split() for x in open(config['file_vocab_x_txt'])]
    vocab_x = PairVocab(vocab_x, cuda=True)
    vocab_atom = VocabAtom(config['COMMON_ATOMS'])
    return vocab_mof, vocab_y, vocab_x, vocab_atom


def get_vecs(model, batch, device):
    graphs, tensors, orders, mol_batch = batch
    tree_tensors, graph_tensors = make_cuda(tensors)
    mol_batch['mof'] = mol_batch['mof'].to(device)
    mol_batch['y'] = mol_batch['y'].to(device)
    mol_batch['y_mask'] = mol_batch['y_mask'].to(device)
    root_vecs, tree_vecs, _, _ = model.encoder(tree_tensors, graph_tensors)
    h_mof = model.enc_mof(mol_batch['mof'])
    root_vecs = root_vecs + h_mof
    root_vecs, _ = model.rsample(root_vecs, model.R_mean, model.R_var, perturb=False)
    return root_vecs, tree_vecs, mol_batch


def invalid_values(len_col_y, n: int) -> Tuple[np.ndarray, np.ndarray]:
    shape = np.ones(len_col_y, )
    invalid_value = -5000 * np.ones_like(shape)
    return np.array([invalid_value] * n, dtype=np.float32), np.zeros(n, dtype=np.float32)


def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).cuda().long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors


def plot_scatter_with_deviation(y_true, y_pred, r2, r2_title):
    # 计算偏差
    y = np.array(y_true)
    x = np.array(y_pred)
    deviation = np.abs(y - x)
    deviation = deviation / deviation.max()
    deviation = np.clip(deviation, np.percentile(deviation, 0), np.percentile(deviation, 80))

    cmap = LinearSegmentedColormap.from_list('custom_cmap',
                                             [(161 / 255, 225 / 255, 250 / 255), (12 / 255, 108 / 255, 196 / 255)])

    plt.figure(figsize=(8, 6.5), dpi=500)
    plt.scatter(y_pred, y_true, alpha=0.5, s=120, c=deviation, cmap=cmap)

    plt.text(0.05, 0.95, f'R² = {r2:.2f}', fontsize=32, color='black', ha='left', va='top',
             transform=plt.gca().transAxes, fontweight='bold')

    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=3)

    plt.xlabel('Predicted values', fontsize=36, fontweight='bold')
    plt.ylabel('Calculated values', fontsize=36, fontweight='bold')
    plt.title(r2_title, fontsize=36, fontweight='bold')

    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    plt.tick_params(width=3, labelsize=4)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca(), label='Deviation value')
    cbar.set_ticks([])
    cbar.ax.yaxis.label.set_size(36)
    cbar.ax.yaxis.label.set_fontweight('bold')

    return plt


def pca_image(data_pca, y, index, y_label):
    plt.figure(figsize=(8, 6.5), dpi=500)
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=y[:, index], s=2.5, lw=1.5, cmap='viridis_r',
                          edgecolor=None)
    plt.colorbar(scatter)
    plt.title(y_label, fontsize=26)
    plt.gca().get_xaxis().set_ticklabels([])  # 隐藏横坐标刻度标签
    plt.gca().get_yaxis().set_ticklabels([])  # 隐藏纵坐标刻度标签
    return plt


def perturb_z(z, noise_norm, constant_norm=False):
    if noise_norm > 0.0:
        noise_vec = np.random.normal(0, 1, size=z.shape).astype(float)
        noise_vec = noise_vec / np.linalg.norm(noise_vec).astype(float)
        if constant_norm:
            return (z + (noise_norm * noise_vec)).float()
        else:
            noise_amp = np.random.uniform(
                0, noise_norm, size=(z.shape[0], 1))
            return (z + (noise_amp * noise_vec)).float()
    else:
        return z.float()


def number_smiles_lr(smiles):
    pattern = re.compile(r'\[Lr]')
    matches = re.findall(pattern, smiles)
    return len(matches)


def capacity_score_smiles(x):
    scorer = SCScorer()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).cuda()
    if number_smiles_lr(x) != 2:
        return False
    else:
        (_, sco) = scorer.get_score_from_smi(x)
        if sco == 0:
            return False
        else:
            return True


def regression_statistics(y_true, y_pred, targets, prefix=''):
    results = []
    for index, col in enumerate(targets):
        result = OrderedDict({'label': col})
        r2 = sklearn.metrics.r2_score(y_true[index], y_pred[index])
        mae = sklearn.metrics.mean_absolute_error(y_true[index], y_pred[index])
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true[index], y_pred[index]))
        result[prefix + 'R2'] = r2
        result[prefix + 'MAE'] = mae
        result[prefix + 'RMSE'] = rmse
        print(col, ': R2 = ', r2, ', MAE = ', mae, ', RMSE = ', rmse)
        results.append(result)

    return pd.DataFrame(results)


def masked_mse_loss(y_hat, y, mask):
    loss_tensor = F.mse_loss(y_hat * mask, y * mask, reduction='none')
    loss = torch.sum(loss_tensor) / (torch.sum(mask) + 1e-7)
    return loss


def masked_wmse_loss(y_hat, y, mask, w):
    # y_hat = torch.squeeze(y_hat, 1)
    loss_tensor = F.mse_loss(y_hat * mask, y * mask, reduction='none')
    weighted_loss = loss_tensor * w
    loss = torch.sum(weighted_loss) / (torch.sum(mask) + 1e-7)
    return loss
