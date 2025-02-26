from collections import OrderedDict, defaultdict, deque
import pandas as pd
import sklearn
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
from rdkit import DataStructs
import rdkit
from rdkit import Chem
import math
import sys
import numpy as np
import os
from tqdm.auto import tqdm
from configs import get_config, AttributeDict
import utils
from generator.dataset import DataFolder
from generator.model import VAE

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


class ConstScheduler:
    def __init__(self, value):
        self.value = value

    def __call__(self, i):
        return self.value


class LinearScheduler:
    def __init__(self, start, end, w_start=0.0, w_end=1.0):
        self.i_start = start
        self.i_end = end
        self.w_start = w_start
        self.w_end = w_end
        self.rate = (self.w_end - self.w_start) / (self.i_end - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        w = min(self.w_start + k * self.rate, self.w_end)
        return w


class CyclicScheduler:
    def __init__(self, cycle_length, const_length, n_epochs, w_start=0.0, w_end=1.0):
        self.cycle_length = cycle_length
        self.const_length = const_length
        self.n_epochs = n_epochs
        n_cycles = np.floor(n_epochs / (cycle_length + const_length))
        self.max_cycle_epoch = n_cycles * (cycle_length + const_length)
        self.w_start = w_start
        self.w_end = w_end

    def __call__(self, i):
        cur_epoch = i % (self.cycle_length + self.const_length)
        growing = cur_epoch < self.cycle_length and i < self.max_cycle_epoch
        if growing:
            rate = (self.w_end - self.w_start)
            t = cur_epoch / self.cycle_length
            return rate * t + self.w_start
        else:
            return self.w_end


def setup_schedulers(config):
    schedulers = OrderedDict()
    n_epochs = config['generator_train_epoch']
    schedulers['x'] = ConstScheduler(1.0)
    schedulers['kl'] = CyclicScheduler(cycle_length=config['kl_cycle_length'],
                                       const_length=config['kl_cycle_constant'],
                                       n_epochs=n_epochs,
                                       w_start=config['kl_weight_start'],
                                       w_end=config['kl_weight_end'])
    schedulers['y'] = LinearScheduler(start=config['y_start'],
                                      end=n_epochs - config['y_const_length'],
                                      w_start=config['y_w_start'],
                                      w_end=config['y_w_end'])
    schedulers['mof'] = LinearScheduler(start=config['mof_start'],
                                        end=n_epochs - config['mof_const_length'],
                                        w_start=config['mof_w_start'],
                                        w_end=config['mof_w_end'])
    return schedulers


class TrainStats:
    def __init__(self, config):
        self.filename = config['TrainStats_filename']
        self.stats = None
        self.report_stats = None
        self.trues = None
        self.preds = None
        self.results = []
        self.buffers = None

    @property
    def report(self):
        return self.report_stats

    def setup_batch_buffers(self, train_n, test_n):
        buffers = OrderedDict()
        for key in config['ALL_COMPONENTS']:
            buffers[f'train_{key}'] = deque(maxlen=train_n)
            buffers[f'test_{key}'] = deque(maxlen=test_n)
        self.buffers = buffers

    def start_epoch(self, epoch, lr, weights):
        self.stats = OrderedDict([('epoch', epoch)])
        self.report_stats = OrderedDict([('lr', lr)])
        self.report_stats.update([(f'λ_{key}', v) for key, v in weights.items()])
        self.preds = defaultdict(list)
        self.trues = defaultdict(list)

    def update_batch(self, prefix, losses, trues=None, preds=None):
        reported = {}
        for key in config['ALL_COMPONENTS']:
            value = losses[key].item()
            self.buffers[f'{prefix}_{key}'].append(value)
            reported[key] = np.mean(self.buffers[f'{prefix}_{key}'])
        if trues and preds:
            for pred_key in preds:
                if preds.get(pred_key) is not None:
                    if type(preds.get(pred_key)) is list:
                        self.preds[pred_key].append(preds[pred_key])
                    else:
                        self.preds[pred_key].append(preds[pred_key].cpu().numpy())
            for true_key in trues:
                if trues.get(true_key) is not None:
                    if type(preds.get(true_key)) is list:
                        self.trues[true_key].append(trues[true_key])
                    else:
                        self.trues[true_key].append(trues[true_key].cpu().numpy())

        return reported

    def update_epoch(self, prefix, reportable=False):
        for key in config['ALL_COMPONENTS']:
            full_key = f'{prefix}_{key}'
            if reportable:
                self.report_stats[full_key] = np.mean(self.buffers[full_key])
            else:
                self.stats[full_key] = np.mean(self.buffers[full_key])

    def compute_metrics(self, model):
        def flat_map(x, inner_fn=lambda x: x, outer_fn=lambda x: x):
            return outer_fn([inner_fn(j) for i in x for j in i])

        if self.preds['x']:
            ms1 = [Chem.MolFromSmiles(j[0]) for i in self.trues['x'] for j in i]
            ms2 = [Chem.MolFromSmiles(j) for i in self.preds['x'] for j in i]  # j[0]与ｊ的区别在于数据格式不同
            fps1 = [Chem.RDKFingerprint(x) for x in ms1]
            fps2 = [Chem.RDKFingerprint(x) for x in ms2]
            similarity = 0
            for i in range(len(fps1)):
                similarity += DataStructs.FingerprintSimilarity(fps1[i], fps2[i])
            self.report_stats['reconstuct_smiles'] = similarity * 1.0 / len(fps1)
            true_smiles = [j[0] for i in self.trues['x'] for j in i]
            pred_smiles = [j for i in self.preds['x'] for j in i]
            same_smiles = [pred_smiles[i] for i in range(len(pred_smiles)) if pred_smiles[i] == true_smiles[i]]
            self.report_stats['same_smiles'] = len(same_smiles) * 1.0 / len(true_smiles)
        if self.preds['mof']:
            mof_hat = flat_map(self.preds['mof'], outer_fn=np.stack)
            mof = flat_map(self.trues['mof'], outer_fn=np.stack)
            samesies = np.sum(np.equal(mof, mof_hat))
            self.report_stats['mof_acc'] = samesies / np.prod(mof.shape) * 100.0
        if self.preds['y']:
            y_trues = flat_map(self.trues['y'], outer_fn=np.stack)
            y_preds = flat_map(self.preds['y'], outer_fn=np.stack)
            mask = flat_map(self.trues['y_mask'], outer_fn=np.stack)
            mask = mask.ravel().astype(bool)
            y_trues = model.vocab_y.inverse_transform(y_trues[mask])
            y_preds = model.vocab_y.inverse_transform(y_preds[mask])
            values = OrderedDict()
            labels = model.vocab_y.labels
            for i, label in enumerate(labels):
                try:
                    values[f'{label}-r2'] = sklearn.metrics.r2_score(y_trues[i], y_preds[i])
                    values[f'{label}-MAE'] = sklearn.metrics.mean_absolute_error(y_trues[i], y_preds[i])
                except:
                    print('Error in computing metrics for y_trues: ', y_trues[i], 'y_preds: ', y_preds[i])
            self.report_stats['mean_r2'] = np.nanmean([values[f'{label}-r2'] for label in labels])
            self.stats.update(values)

    def finalize_epoch(self, save=True):
        self.stats.update(self.report_stats)
        for prefix in ['train', 'test']:
            loss = self.stats[f'{prefix}_loss']
            for key in config['COMPONENTS']:
                key_loss = self.stats[f'λ_{key}'] * self.stats[f'{prefix}_{key}']
                self.stats[f'{prefix}_{key}_ratio'] = key_loss / loss

        self.results.append(self.stats)
        if save:
            pd.DataFrame(self.results).to_csv(self.filename, index=False)


if __name__ == "__main__":
    config: AttributeDict = get_config()
    utils.set_seed(config['generator_rand_seed'])

    vocab_mof, vocab_y, vocab_x, vocab_atom = utils.set_vocab(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(config, vocab_x, vocab_mof, vocab_y, vocab_atom).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.ExponentialLR(optimizer, config['anneal_rate'])

    if config['load_model']:
        print('---------------- continuing -------------------------------------')
        model_state, optimizer_state, total_step, beta = torch.load(config['best_model'])
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    else:
        total_step = beta = 0

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    meters = {
        'kl': 0.0,  # KL 散度损失
        'loss': 0.0,  # 总损失
        'wacc': 0.0,  # 单词准确率
        'iacc': 0.0,  # 内部准确率
        'tacc': 0.0,  # 拓扑准确率
        'sacc': 0.0  # 组装准确率
    }
    schedulers = setup_schedulers(config)
    stats = TrainStats(config)

    pbar = tqdm(range(config['generator_train_epoch']), desc='Epochs')
    for epoch in pbar:
        epoch += 1
        train_dataset = DataFolder(config['tensors_train'], config['train_batch_size'])
        test_dataset = DataFolder(config['tensors_test'], config['train_batch_size'])

        weights = {key: sch(epoch) for key, sch in schedulers.items()}
        stats.setup_batch_buffers(len(train_dataset), len(test_dataset))
        stats.start_epoch(epoch, config['lr'], weights)

        for batch in tqdm(train_dataset):
            total_step += 1
            model_loss, wacc, iacc, tacc, sacc, _ = model(*batch, beta=beta)
            w_loss = {k: weights[k] * model_loss[k] for k in config['COMPONENTS']}
            w_loss['mof'] = w_loss['mof'] * 2
            loss = w_loss['x'] + beta * w_loss['kl'] + w_loss['mof'] + w_loss['y']
            model_loss['loss'] = loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
            optimizer.step()
            meters['kl'] += model_loss['kl'].item()
            meters['loss'] += loss.item()
            meters['wacc'] += (wacc * 100).cpu()
            meters['iacc'] += (iacc * 100).cpu()
            meters['tacc'] += (tacc * 100).cpu()
            meters['sacc'] += (sacc * 100).cpu()
            pbar.set_postfix(stats.update_batch('train', model_loss))

            if total_step % 50 == 0:
                for key in meters:
                    meters[key] /= 50
                print(
                    "[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, "
                    "GNorm: %.2f, loss_mof: %.2f, loss_y: %.2f, weights: %.2f"
                    % (total_step, beta, meters['kl'], meters['loss'], meters['wacc'], meters['iacc'], meters['tacc'],
                       meters['sacc'],
                       param_norm(model), grad_norm(model), w_loss['mof'], w_loss['y'], weights['mof']))
                print("(weights['mof']:%.2f)*(model_loss['mof']: %.2f) =(w_loss['mof']: %.2f)" %
                      (weights['mof'], model_loss['mof'], w_loss['mof']))
                print("kl_loss: %.2f    mof_loss: %.2f    x_loss: %.2f    y_loss: %.2f    loss: %.2f"
                      % (model_loss['kl'], model_loss['mof'], model_loss['x'], model_loss['y'], model_loss['loss']))
                sys.stdout.flush()
                for key in meters:
                    meters[key] = 0.0

            if total_step % 2000 == 0:
                ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
                torch.save(ckpt, os.path.join(config['train_save_dir'], f"model.ckpt.{total_step}"))
                # torch.save(ckpt, config['best_model'])

            if total_step % 2000 == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_step >= config['warmup'] and total_step % config['kl_anneal_iter'] == 0:
                beta = min(config['max_beta'], beta + config['step_beta'])
        stats.update_epoch('train', reportable=True)
        model.eval()  # test
        torch.set_grad_enabled(False)
        print("----------------test begin -------------------------------------")
        for batch in tqdm(test_dataset):
            model.zero_grad()
            model_loss, wacc, iacc, tacc, sacc, outs = model(*batch, beta=beta)
            _, tensors, _, _ = batch
            outs['x'] = model.reconstruct(tensors)
            _, _, _, mol_batch = batch
            w_loss = {k: weights[k] * model_loss[k] for k in config['COMPONENTS']}
            loss = w_loss['x'] + beta * w_loss['kl'] + w_loss['mof'] + w_loss['y']
            model_loss['loss'] = loss
            trues = {key: mol_batch[key] for key in ['x', 'mof', 'y', 'y_mask']}
            stats.update_batch('test', model_loss, trues, outs)
            stats.update_batch('test', model_loss)
        print("----------------test end -------------------------------------")
        torch.set_grad_enabled(True)
        stats.update_epoch('test', reportable=False)
        stats.compute_metrics(model)
        pbar.set_postfix(stats.report)
        stats.finalize_epoch()
