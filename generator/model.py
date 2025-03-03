from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Text, Any, Tuple
from generator.dec_x import HierMPNDecoder
from generator.enc_x import HierMPNEncoder
from generator.modules import make_mlp
from generator.vocab_mof import MOFVocab
from utils import make_cuda, masked_mse_loss, masked_wmse_loss

tensor = torch.tensor


class VAE(nn.Module):

    def __init__(self, config, vocab_x, vocab_mof, vocab_y, vocab_atom):
        super().__init__()
        self.config = config
        self.vocab_x = vocab_x
        self.vocab_atom = vocab_atom
        self.vocab_mof = vocab_mof
        self.vocab_y = vocab_y
        self.latent_size = config['latent_size']

        self.encoder = HierMPNEncoder(self.vocab_x, self.vocab_atom, config['rnn_type'],
                                      config['embed_size'], config['hidden_size'],
                                      config['depthT'], config['depthG'],
                                      config['dropout'])
        self.decoder = HierMPNDecoder(self.vocab_x, self.vocab_atom,
                                      config['rnn_type'], config['embed_size'],
                                      config['hidden_size'], config['latent_size'],
                                      config['diterT'], config['diterG'],
                                      config['dropout'])
        self.encoder.tie_embedding(self.decoder.hmpn)

        self.R_mean = nn.Linear(config['hidden_size'], config['latent_size'])
        self.R_var = nn.Linear(config['hidden_size'], config['latent_size'])

        self.enc_mof = MOFEncoder(latent_dim=288, hidden_size=config['hidden_size'],
                                  mof_dims=vocab_mof.dims, n_layers=2, act='relu',
                                  batchnorm=False, dropout=config['dropout'])
        self.dec_mof = MOFDecoder(latent_dim=288, latent_size=config['latent_size'],
                                  vocab_mof=vocab_mof, wloss=True, n_layers=1,
                                  act='relu', batchnorm=False, dropout=config['dropout'])
        self.dec_y = PropDecoder(latent_dim=288, latent_size=config['latent_size'],
                                 weights=vocab_y.weights, scaler=vocab_y.scaler,
                                 n_layers=1, act='relu',
                                 batchnorm=False, dropout=0.0)

    def rsample(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def sample(self, batch_size, greedy):
        root_vecs = torch.randn(batch_size, self.latent_size).cuda()
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)

    def reconstruct(self, batch):
        tensors = batch
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb=False)
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)

    def forward(self, graphs, tensors, orders, mol_batch, beta, perturb_z=True):
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)
        loss = {key: torch.tensor(0.0) for key in self.config['COMPONENTS']}
        outs = {'mof': None, 'y': None, 'y_mask': None}
        # 只用了root_vecs这个最底层的嵌入
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mol_batch['mof'] = mol_batch['mof'].to(device)
        mol_batch['y'] = mol_batch['y'].to(device)
        mol_batch['y_mask'] = mol_batch['y_mask'].to(device)

        h_mof = self.enc_mof(mol_batch['mof'])
        root_vecs = root_vecs + h_mof
        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb_z)
        loss['kl'] = root_kl

        loss['x'], wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)

        loss['mof'], outs['mof'] = self.dec_mof(mol_batch['mof'], root_vecs)

        loss['y'], outs['y'] = self.dec_y(root_vecs, mol_batch['y'], mol_batch['y_mask'])
        outs['y_mask'] = mol_batch['y_mask']
        return loss, wacc, iacc, tacc, sacc, outs


class MOFEncoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int, mof_dims: List[int], n_layers: int, act: Text,
                 batchnorm: bool, dropout: float):
        super().__init__()
        self.feat_embs = nn.ModuleList([nn.Embedding(n, latent_dim) for n in mof_dims])
        self.n_features = len(mof_dims)
        self.mof_dims = mof_dims
        self.mlp = make_mlp([latent_dim] * (n_layers - 1) + [latent_dim * 2] + [hidden_size], act, batchnorm, dropout,
                            activation_last=True)

    def forward(self, mof: tensor) -> tensor:
        h = []
        for index, emb in enumerate(self.feat_embs):
            h.append(emb(mof[:, index]))
        h = torch.sum(torch.stack(h), dim=0)
        h = self.mlp(h)
        return h


class MOFDecoder(nn.Module):
    def __init__(self, latent_dim: int, latent_size: int, vocab_mof: MOFVocab, wloss: bool,
                 n_layers: int, act: Text, batchnorm: bool, dropout: float):
        super().__init__()
        self.vocab_mof = vocab_mof
        self.wloss = wloss
        self.n_features = len(vocab_mof.dims)
        self.mof_dims = vocab_mof.dims
        self.weights = [torch.from_numpy(w.astype(np.float32)) for w in vocab_mof.weight_list]
        self.mof_weights = nn.ParameterList([nn.Parameter(w, requires_grad=False) for w in self.weights])
        self.mlp = make_mlp([latent_size] + [latent_dim] * (n_layers - 1) + [latent_dim * 2], act, batchnorm, dropout,
                            activation_last=True)
        self.out_to_id = nn.ModuleList([nn.Linear(latent_dim * 2, i) for i in vocab_mof.dims])

    def forward(self, mof: tensor, z: tensor) -> Tuple[tensor, tensor]:
        out = self.mlp(z)
        mof_hat = []
        loss = 0.0
        for index, cat_map in enumerate(self.out_to_id):
            w = self.mof_weights[index] if self.wloss else None
            target = mof[:, index]
            output = cat_map(out)
            mof_hat.append(torch.argmax(F.softmax(output, dim=-1), dim=1))
            loss += F.cross_entropy(output, target, weight=w)
        mof_hat = torch.transpose(torch.stack(mof_hat), 0, 1)
        return loss, mof_hat

    def z_to_mof(self, z: tensor):
        with torch.no_grad():
            out = self.mlp(z)
            mof_ids = []
            for index, cat_map in enumerate(self.out_to_id):
                output = cat_map(out)
                ids = torch.argmax(F.softmax(output, dim=-1), dim=1)
                mof_ids.append(ids.detach().cpu().numpy())
            mof_ids_list = np.stack(mof_ids).T.tolist()
        mofs = [self.vocab_mof.ids_to_mof(mof) for mof in mof_ids_list]
        return mofs


class PropDecoder(nn.Module):
    def __init__(self, latent_dim: int, latent_size: int, weights: List[float], scaler: Any,
                 n_layers: int, act: Text, batchnorm: bool, dropout: float):
        super().__init__()
        output_dim = len(weights)
        self.scaler = scaler
        weights = torch.from_numpy(np.array(weights).astype(np.float32))
        self.loss_weights = nn.Parameter(weights, requires_grad=False)
        layer_dims = [latent_size] + [latent_dim] * (n_layers - 1) + [output_dim]
        self.mlp = make_mlp(layer_dims, act, batchnorm, dropout, activation_last=False)

    def forward(self, z: tensor, y: tensor, mask: tensor) -> Tuple[tensor, tensor]:
        y_hat = self.mlp(z)
        loss = masked_wmse_loss(y_hat, y, mask, self.loss_weights)
        return loss, y_hat

    def z_to_y(self, z: tensor) -> np.ndarray:
        with torch.no_grad():
            y_hat = self.mlp(z).cpu().numpy()
        y_hat = self.scaler.inverse_transform(y_hat)
        return y_hat

    def z_to_scalar_y(self, z: tensor) -> np.ndarray:
        with torch.no_grad():
            y_hat = self.mlp(z).cpu().numpy()
        return y_hat


class PropDecoderWithMotif(nn.Module):
    def __init__(self, latent_dim: int, latent_size: int, vocab_y,
                 n_layers: int, act: Text, batchnorm: bool, dropout: float):
        super().__init__()
        weights = vocab_y.weights
        self.scaler = vocab_y.scaler
        self.labels = vocab_y.labels
        weights = torch.from_numpy(np.array(weights).astype(np.float32))
        self.loss_weights = nn.Parameter(weights, requires_grad=False)

        layer_dims = ([314] + [latent_size] + [latent_dim] * (n_layers + 2)
                      + [int(latent_dim / 2)] + [int(latent_dim / 4)])
        self.mlp = make_mlp(layer_dims, act, batchnorm, dropout, activation_last=False)

        mlp_spilt_dims = [int(latent_dim / 4)] + [int(latent_dim / 8)] + [1]
        self.mlp_lcd = make_mlp(mlp_spilt_dims, act, batchnorm, dropout, activation_last=False)
        self.mlp_pld = make_mlp(mlp_spilt_dims, act, batchnorm, dropout, activation_last=False)
        self.mlp_density = make_mlp(mlp_spilt_dims, act, batchnorm, dropout, activation_last=False)
        self.mlp_agsa = make_mlp(mlp_spilt_dims, act, batchnorm, dropout, activation_last=False)
        self.mlp_co2n2_co2_mol_kg = make_mlp(mlp_spilt_dims, act, batchnorm, dropout, activation_last=False)
        self.mlp_co2n2_n2_mol_kg = make_mlp(mlp_spilt_dims, act, batchnorm, dropout, activation_last=False)
        self.mlp_co2ch4_co2_mol_kg = make_mlp(mlp_spilt_dims, act, batchnorm, dropout, activation_last=False)
        self.mlp_co2ch4_ch4_mol_kg = make_mlp(mlp_spilt_dims, act, batchnorm, dropout, activation_last=False)

        self.mlp_spilt = [self.mlp_lcd, self.mlp_pld, self.mlp_density, self.mlp_agsa, self.mlp_co2n2_co2_mol_kg,
                          self.mlp_co2n2_n2_mol_kg, self.mlp_co2ch4_co2_mol_kg, self.mlp_co2ch4_ch4_mol_kg]

    def forward(self, root_vecs: torch.Tensor, tree_vecs: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) \
            -> tuple[list, torch.Tensor]:
        tree_vecs_pooled = tree_vecs.mean(dim=0)
        tree_vecs_pooled_transformed = tree_vecs_pooled.view(1, -1).expand(root_vecs.size(0), -1)
        concatenated_features = torch.cat((root_vecs, tree_vecs_pooled_transformed), dim=1)
        y_hat = OrderedDict()
        concatenated_features = self.mlp(concatenated_features)
        for index, net in enumerate(self.mlp_spilt):
            y_hat[index] = net(concatenated_features)
        y_pred = torch.cat([y_hat[i] for i in y_hat], dim=1)
        losses = [masked_mse_loss(y_pred[:, index], y[:, index], mask) for index in range(len(self.labels))]
        return losses, y_pred

    def z_to_y(self, root_vecs: torch.Tensor, tree_vecs: torch.Tensor) -> np.ndarray:
        y_pred = self.z_to_scalar_y(root_vecs, tree_vecs)
        y = self.scaler.inverse_transform(y_pred.cpu().numpy())
        return y

    def z_to_scalar_y(self, root_vecs: torch.Tensor, tree_vecs: torch.Tensor) -> torch.Tensor:
        y_hat = OrderedDict()
        tree_vecs_pooled = tree_vecs.mean(dim=0)
        tree_vecs_pooled_transformed = tree_vecs_pooled.view(1, -1).expand(root_vecs.size(0), -1)
        concatenated_features = torch.cat((root_vecs, tree_vecs_pooled_transformed), dim=1)

        concatenated_features = self.mlp(concatenated_features)
        with torch.no_grad():
            for index, net in enumerate(self.mlp_spilt):
                y_hat[index] = net(concatenated_features)
            y_pred = torch.cat([y_hat[i] for i in y_hat], dim=1)
        return y_pred
