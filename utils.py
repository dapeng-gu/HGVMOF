import pickle
import random
import re
from collections import OrderedDict
from typing import Tuple
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sklearn.metrics
import pandas as pd
from sklearn.model_selection import KFold
from torch.nn import functional as F
from more_itertools import chunked
from tqdm.autonotebook import tqdm

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

    plt.figure(figsize=(8, 6.5), dpi=100)
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
    plt.figure(figsize=(8, 6.5), dpi=100)
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


def sample_model(inference_model, n, decice, batch_size=50, smiles_column='branch_smiles'):
    n_loops = int(np.ceil(n / batch_size))
    gen_df = pd.DataFrame()
    z_list = []
    for chunk in tqdm(chunked(range(n), batch_size), total=n_loops, desc='Samples'):
        z = torch.randn(len(chunk), 64, device=decice)
        z_list.append(z)
        mof_building = inference_model.mof_z_to_mof_building(z)
        mof_y = inference_model.mof_z_to_mof_y(z)
        outs = inference_model.mof_building_and_mof_y_to_mof_dict(mof_building, mof_y)
        if gen_df.empty:
            gen_df = pd.DataFrame(outs)
        else:
            gen_df = pd.concat([gen_df, pd.DataFrame(outs)])
    gen_df['valid'] = gen_df[smiles_column].apply(capacity_score_smiles)

    return gen_df.reset_index(drop=True), torch.cat(z_list, dim=0)


def plot_scatter_with_deviation_and_ci_new(y_true, y_pred, r2, r2_title, index, n_bootstrap=1000, k_folds=5, ci=0.95):

    _, ci_lower, ci_upper = combined_r2_ci(y_true, y_pred, n_bootstrap=n_bootstrap, k_folds=k_folds, ci=ci)
    
    # 计算偏差
    y = np.array(y_true)
    x = np.array(y_pred)
    deviation = np.abs(y - x)
    deviation = deviation / deviation.max()
    deviation = np.clip(deviation, np.percentile(deviation, 0), np.percentile(deviation, 80))

    cmap = LinearSegmentedColormap.from_list('custom_cmap',
                                             [(161 / 255, 225 / 255, 250 / 255), (12 / 255, 108 / 255, 196 / 255)])

    plt.figure(figsize=(8, 6.5), dpi=100)
    plt.scatter(y_pred, y_true, alpha=0.5, s=120, c=deviation, cmap=cmap)

    # 添加R2和置信区间文本（分两行，置信区间字体较小）
    ci_percent = int(ci * 100)
    plt.text(0.05, 0.95, f'R² = {r2:.2f}', fontsize=30, color='black', ha='left', va='top',
             transform=plt.gca().transAxes, fontweight='bold')
    plt.text(0.05, 0.86, f'({ci_percent}% CI: [{ci_lower:.2f}, {ci_upper:.2f}])',
             fontsize=20, color='black', ha='left', va='top',
             transform=plt.gca().transAxes, fontweight='bold')

    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=3)

    plt.xlabel('Predicted values', fontsize=30, fontweight='bold')
    plt.ylabel('Calculated values', fontsize=30, fontweight='bold')
    plt.title(r2_title, fontsize=30, fontweight='bold')

    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    plt.tick_params(width=3, labelsize=4)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca(), label='Deviation value')
    cbar.set_ticks([])
    cbar.ax.yaxis.label.set_size(30)
    cbar.ax.yaxis.label.set_fontweight('bold')
    plt.savefig(f'results/images/' + str(index) + '.png')

    return plt


def bootstrap_r2_ci(y_true, y_pred, n_iterations=1000, ci=0.95):
    r2_scores = []
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算原始R2
    orig_r2 = r2_score(y_true, y_pred)

    for _ in range(n_iterations):
        indices = np.random.choice(range(n), size=n, replace=True)
        if len(np.unique(y_true[indices])) > 1:  # 确保有足够的变异性
            r2 = r2_score(y_true[indices], y_pred[indices])
            r2_scores.append(r2)

    # 计算置信区间
    lower = np.percentile(r2_scores, (1 - ci) / 2 * 100)
    upper = np.percentile(r2_scores, (1 + ci) / 2 * 100)
    mean_r2 = np.mean(r2_scores)

    return mean_r2, lower, upper


def cv_r2_ci(X, y, k=5, ci=0.95):
    """
    使用交叉验证方法计算R2的置信区间

    参数:
    X: 特征矩阵
    y: 目标变量
    k: 折数
    ci: 置信水平 (0-1)

    返回:
    mean_r2: 平均R2值
    lower: 置信区间下限
    upper: 置信区间上限
    """
    X = np.array(X).reshape(-1, 1) if len(np.array(X).shape) == 1 else np.array(X)
    y = np.array(y)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_values = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_values.append(r2)

    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)

    # 使用t分布计算置信区间
    t_value = stats.t.ppf((1 + ci) / 2, k - 1)
    margin = t_value * (std_r2 / np.sqrt(k))

    lower = max(0, mean_r2 - margin)  # 确保R2不小于0
    upper = min(1, mean_r2 + margin)  # 确保R2不大于1

    return mean_r2, lower, upper


def combined_r2_ci(y_true, y_pred, n_bootstrap=1000, k_folds=5, ci=0.95):
    """
    结合Bootstrap和交叉验证方法计算R2的置信区间

    参数:
    y_true: 真实值数组
    y_pred: 预测值数组
    n_bootstrap: Bootstrap迭代次数
    k_folds: 交叉验证折数
    ci: 置信水平 (0-1)

    返回:
    r2: 原始R2值
    lower: 置信区间下限
    upper: 置信区间上限
    """
    # 计算原始R2
    r2 = r2_score(y_true, y_pred)

    # Bootstrap置信区间
    bs_mean, bs_lower, bs_upper = bootstrap_r2_ci(y_true, y_pred, n_iterations=n_bootstrap, ci=ci)

    # 为交叉验证准备数据
    X = np.array(y_true).reshape(-1, 1)  # 使用真实值作为特征
    y = np.array(y_pred)  # 使用预测值作为目标

    # 交叉验证置信区间 (反转X和y是为了评估预测的一致性)
    cv_mean, cv_lower, cv_upper = cv_r2_ci(X, y, k=k_folds, ci=ci)

    # 综合两种方法
    final_lower = max(0, (bs_lower + cv_lower) / 2)  # 确保R2不小于0
    final_upper = min(1, (bs_upper + cv_upper) / 2)  # 确保R2不大于1

    return r2, final_lower, final_upper
