import os
from collections import Counter, OrderedDict


class AttributeDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_namespace(cls, namespace):
        return cls(vars(namespace))


def get_config() -> AttributeDict:
    config = AttributeDict()
    config.update(basic_config())
    config.update(data_config())
    config.update(y_config())
    config.update(hardware_config())
    config.update(file_site_config())
    config.update(train_config())
    config.update(generator_config())
    config.update(vocab_atom_config())
    config.update(load_config())
    config.update(opt_config())
    return config


def basic_config():
    config = AttributeDict()
    config['test_start'] = False
    return config


def data_config():
    config = AttributeDict()
    config['df_csv_file'] = 'data/original_data/MOF_gen_train.csv.gz'
    config['df_prop_csv_file'] = 'data/original_data/MOF_properties_train.csv'
    config['scscore_WEIGHTS_FILE'] = 'data/original_data/scscore_1024uint8_model.ckpt-10654.as_numpy.json.gz'

    config['col_x'] = 'branch_smiles'
    config['col_mof'] = ['metal_node', 'organic_core', 'topology']
    config['col_y'] = ['lcd', 'pld', 'density', 'agsa', 'co2n2_co2_mol_kg',
                       'co2n2_n2_mol_kg', 'co2ch4_co2_mol_kg', 'co2ch4_ch4_mol_kg']
    config['col_testtrain'] = 'train/test'
    return config


def y_config():
    config = AttributeDict()
    config['y_weights'] = [1, 1, 1, 1, 1, 1, 1, 1]
    config['y_scaler_type'] = 'standard'
    return config


def hardware_config():
    config = AttributeDict()
    config['cpu_cores_num'] = 7
    return config


def file_site_config():
    config = AttributeDict()
    config['file_df_train'] = 'data/df/df_train.pkl'
    config['file_df_test'] = 'data/df/df_test.pkl'
    config['file_df_prop_train'] = 'data/df/df_prop_train.pkl'
    config['file_df_prop_test'] = 'data/df/df_prop_test.pkl'
    config['file_ids2mof'] = 'data/df/ids2mof.pkl'
    config['file_mof2ids'] = 'data/df/mof2ids.pkl'
    config['tensors_train'] = 'data/tensors/train'
    config['tensors_test'] = 'data/tensors/test'
    config['tensors_train_prop'] = 'data/tensors/train_prop'
    config['tensors_test_prop'] = 'data/tensors/test_prop'

    config['file_vocab_x_txt'] = 'results/vocabs/vocab_x.txt'
    config['file_vocab_mof'] = 'results/vocabs/vocab_mof.pkl'
    config['file_vocab_y'] = 'results/vocabs/vocab_y.pkl'

    config['files_log'] = 'results/log/log.csv'
    config['train_save_dir'] = 'results/train_model'
    config['train_y_save_dir'] = 'results/train_model_y'

    config['TrainStats_filename'] = 'results/train_stats/log.csv'

    config['r2_image_file_dir'] = 'results/images/r2/'

    config['file_opt_best_solution'] = 'results/opt_log/best_solution/'
    config['file_opt_best_fitness_history'] = 'results/opt_log/best_fitness_history/'
    config['file_opt_best_step_mof_history'] = 'results/opt_log/best_step_mof_history/'

    return config


def train_config():
    config = AttributeDict()
    config['train_batch_size'] = 50
    config['get_tensors_batch_size'] = 1536
    config['generator_rand_seed'] = 7
    config['generator_train_epoch'] = 120
    config['predic_train_epoch'] = 200
    return config


def generator_config():
    config = AttributeDict()

    config['COMPONENTS'] = ['x', 'mof', 'y', 'kl']
    config['ALL_COMPONENTS'] = ['x', 'mof', 'y', 'kl', 'loss']

    config['clip_norm'] = 5.0
    config['lr'] = 1e-3

    config['rnn_type'] = 'LSTM'
    config['embed_size'] = 250
    config['hidden_size'] = 250
    config['latent_size'] = 64
    config['depthT'] = 15
    config['depthG'] = 15
    config['diterT'] = 1
    config['diterG'] = 3
    config['dropout'] = 0.0

    config['anneal_rate'] = 0.9
    config['warmup'] = 10000
    config['kl_anneal_iter'] = 2000

    config['step_beta'] = 0.001
    config['max_beta'] = 1.0

    config["kl_cycle_length"] = 15
    config["kl_cycle_constant"] = 3
    config["kl_weight_start"] = 1e-5
    config["kl_weight_end"] = 0.002794217

    config['mof_weighted_loss'] = True
    config['mof_w_start'] = 0.0001
    config['mof_w_end'] = 0.1
    config['mof_start'] = 0
    config['mof_const_length'] = 10

    config['y_w_start'] = 0.0001
    config['y_w_end'] = 0.1
    config['y_start'] = 0
    config['y_const_length'] = 10
    return config


def vocab_atom_config():
    config = AttributeDict()
    config['COMMON_ATOMS'] = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1),
                              ('C', -1), ('Cl', 0), ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0),
                              ('F', 1), ('F', -1), ('I', -1), ('I', 0), ('I', 1), ('I', 2), ('I', 3), ('N', 0),
                              ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1), ('P', 0), ('P', 1), ('P', -1),
                              ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0), ('Si', -1),
                              ('Lr', 0)]

    return config


def load_config():
    config = AttributeDict()
    config['load_model'] = True
    config['load_model_y'] = False
    config['best_model'] = 'results/best/model'
    config['best_model_y'] = 'results/best/model_y'
    return config


def opt_config():
    config = AttributeDict()
    config['opt_num_part'] = 10
    config['opt_num_swarms'] = 1
    config['opt_iterations_num'] = 20
    config['opt_num_track'] = 10
    # values in ['lcd', 'pld','density','agsa', 'co2n2_co2_mol_kg',
    #            'co2n2_n2_mol_kg', 'co2ch4_co2_mol_kg','co2ch4_ch4_mol_kg']
    config['opt_fitness_name'] = 'co2n2_co2_mol_kg'

    config['opt_col'] = ['fitness',
                         'branch_smiles', 'metal_node', 'organic_core', 'topology',
                         'lcd', 'pld', 'density', 'agsa',
                         'co2n2_co2_mol_kg', 'co2n2_n2_mol_kg', 'co2ch4_co2_mol_kg', 'co2ch4_ch4_mol_kg']

    config['x_min'] = 'results/z_max_min/x_min.npy'
    config['x_max'] = 'results/z_max_min/x_max.npy'
    config['v_min'] = -0.4
    config['v_max'] = 0.4
    return config
