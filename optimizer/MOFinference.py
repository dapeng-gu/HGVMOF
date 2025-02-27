import copy
import pickle
import pandas as pd

import torch

import utils
from generator.model import VAE, PropDecoderWithMotif, make_cuda
from preprocess import graphs_to_tuples, tensorize


class InferenceModel(object):
    def __init__(self, config):
        self.config = config
        self.vocab_mof, self.vocab_y, self.vocab_x, self.vocab_atom = utils.set_vocab(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_state, _, _, _ = torch.load(config['best_model'])
        model_y_state, _, _ = torch.load(config['best_model_y'])

        self.model = VAE(self.config, self.vocab_x, self.vocab_mof, self.vocab_y, self.vocab_atom).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()

        self.model_y = PropDecoderWithMotif(latent_dim=config['predict_latent_dim'],
                                            latent_size=config['predict_latent_size'],
                                            vocab_y=self.vocab_y, n_layers=config['n_layers'],
                                            act='relu', batchnorm=False,
                                            dropout=0.0).to(self.device)
        self.model_y.load_state_dict(model_y_state)
        self.model_y.eval()

    def mof_building_to_mof_tensor(self, mof_building):
        df_mof_building = pd.DataFrame(mof_building,
                                       columns=['organic_core', 'metal_node', 'topology', 'branch_smiles'])
        df_tuple = graphs_to_tuples(df_mof_building, self.config['col_x'], self.vocab_mof, self.vocab_y)
        return tensorize(df_tuple, self.vocab_x, self.vocab_atom)

    def mof_tensor_to_mof_z(self, mof_tensor):
        with torch.no_grad():
            root_vecs, _, _ = utils.get_vecs(self.model, mof_tensor, self.device)
        return root_vecs

    def mof_tensor_to_mof_y(self, mof_tensor):
        model_y = self.model_y
        with torch.no_grad():
            root_vecs, tree_vecs, _ = utils.get_vecs(self.model, mof_tensor, self.device)
            outputs = model_y.z_to_y(root_vecs, tree_vecs)
        return outputs

    def mof_z_to_mof_building(self, mof_z):
        model = self.model
        out = {'x': None, 'mof': None, 'y': None}
        mof_building_list = []
        with torch.no_grad():
            mof_z = mof_z.to(self.device)
            out['x'] = model.decoder.decode((mof_z, mof_z, mof_z), greedy=True, max_decode_step=150)
            out['mof'] = model.dec_mof.z_to_mof(mof_z)
            for index in range(len(out['x'])):
                mof_building = (out['mof'][index][1], out['mof'][index][0], out['mof'][index][2], out['x'][index])
                mof_building_list.append(mof_building)
        return mof_building_list

    def mof_z_to_mof_y(self, mof_z):
        mof_building = self.mof_z_to_mof_building(mof_z)
        mof_tensor = self.mof_building_to_mof_tensor(mof_building)
        mof_y = self.mof_tensor_to_mof_y(mof_tensor)
        return mof_y

    @staticmethod
    def mof_building_and_mof_y_to_mof_dict(mof_building, mof_y):
        mof_dict = {
            'organic_core': None,
            'metal_node': None,
            'topology': None,
            'branch_smiles': None,
            'lcd': 0,
            'pld': 0,
            'density': 0,
            'agsa': 0,
            'co2n2_co2_mol_kg': 0,
            'co2n2_n2_mol_kg': 0,
            'co2ch4_co2_mol_kg': 0,
            'co2ch4_ch4_mol_kg': 0
        }
        mof_dict_list = []
        for index in range(len(mof_building)):
            new_mof_dict = copy.deepcopy(mof_dict)
            new_mof_dict['organic_core'] = mof_building[index][0]
            new_mof_dict['metal_node'] = mof_building[index][1]
            new_mof_dict['topology'] = mof_building[index][2]
            new_mof_dict['branch_smiles'] = mof_building[index][3]

            new_mof_dict['lcd'] = mof_y[index][0]
            new_mof_dict['pld'] = mof_y[index][1]
            new_mof_dict['density'] = mof_y[index][2]
            new_mof_dict['agsa'] = mof_y[index][3]
            new_mof_dict['co2n2_co2_mol_kg'] = mof_y[index][4]
            new_mof_dict['co2n2_n2_mol_kg'] = mof_y[index][5]
            new_mof_dict['co2ch4_co2_mol_kg'] = mof_y[index][6]
            new_mof_dict['co2ch4_ch4_mol_kg'] = mof_y[index][7]

            mof_dict_list.append(new_mof_dict)
        return mof_dict_list
