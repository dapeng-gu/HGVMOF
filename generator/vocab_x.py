
import torch
from rdkit import Chem


class PairVocab(object):

    def __init__(self, smiles_pairs, cuda=True):
        cls = list(zip(*smiles_pairs))[0]
        self.hvocab = sorted(list(set(cls)))
        self.hmap = {x: i for i, x in enumerate(self.hvocab)}

        self.vocab = [tuple(x) for x in smiles_pairs]  # copy
        self.inter_size = [count_inters(x[1]) for x in self.vocab]
        self.vmap = {x: i for i, x in enumerate(self.vocab)}

        self.mask = torch.zeros(len(self.hvocab), len(self.vocab))
        for h, s in smiles_pairs:
            hid = self.hmap[h]
            idx = self.vmap[(h, s)]
            self.mask[hid, idx] = 1000.0

        if cuda:
            self.mask = self.mask.cuda()
        self.mask = self.mask - 1000.0

    def __getitem__(self, x):
        assert type(x) is tuple
        try:
            return self.hmap[x[0]], self.vmap[x]
        except:
            return 0, 0

    def get_smiles(self, idx):
        return self.hvocab[idx]

    def get_ismiles(self, idx):
        return self.vocab[idx][1]

    def size(self):
        return len(self.hvocab), len(self.vocab)

    def get_mask(self, cls_idx):
        return self.mask.index_select(index=cls_idx, dim=0)

    def get_inter_size(self, icls_idx):
        return self.inter_size[icls_idx]


def count_inters(s):
    mol = Chem.MolFromSmiles(s)
    inters = [a for a in mol.GetAtoms() if a.GetAtomMapNum() > 0]
    return max(1, len(inters))
