class VocabAtom(object):

    def __init__(self, smiles_list):
        self.vocab = [x for x in smiles_list]
        self.vmap = {x: i for i, x in enumerate(self.vocab)}

    def __getitem__(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def size(self):
        return len(self.vocab)
