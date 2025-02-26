from collections import OrderedDict
import numpy as np
from more_itertools import chunked
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


class MOFVocab:
    @classmethod
    def from_data(cls, df, columns, weighting=True):
        encoders = OrderedDict()
        weights = OrderedDict()
        for col in columns:
            enc = LabelEncoder()            
            enc.fit(df[col].values)                    
            values = df[col].tolist()
            mapping = dict(zip(enc.classes_, range(len(enc.classes_))))   
            labels, counts = np.unique(values, return_counts=True)      
            max_count = np.max(counts)  
            w = np.zeros(len(enc.classes_), dtype=np.float32)
            for label, count in zip(labels, counts):
                if weighting:
                    w[mapping[label]] = max_count / count
                else:
                    w[mapping[label]] = 1.0

            weights[col] = w
            encoders[col] = enc                     
        print(f'Used columns ={columns} with frequency weighting={weighting}')
        for col in columns:
            print(f'{col:12s} has {len(encoders[col].classes_)} classes')
        return cls(encoders, weights)

    def __init__(self, encoders, weights):

        self.categories = list(encoders.keys())
        self.weights = weights
        self.weight_list = [w for w in weights.values()]
        self.encoders = encoders
        self.n_encoders = len(encoders)
        self.dims = [len(enc.classes_) for enc in encoders.values()]
        self.total_dim = sum(self.dims)

    def __len__(self):
        return sum(self.dims)

    def get_label_to_id(self, key):
        enc = self.encoders[key]
        return dict(zip(enc.classes_, range(len(enc.classes_))))

    def get_id2label(self, key):
        enc = self.encoders[key]
        return dict(zip(range(len(enc.classes_)), enc.classes_))

    def df_to_ids(self, df, batch=10000):
        n = len(df)
        mof_ids = np.zeros((n, self.n_encoders), dtype=int)
        n_loops = int(np.ceil(len(df) / batch))
        disable = n_loops < 5
        for indexes in tqdm(chunked(range(n), batch), total=n_loops, disable=disable, desc='MOF'):
            sub_df = df.iloc[list(indexes)]
            for index, (col, enc) in enumerate(self.encoders.items()):
                mof_ids[indexes, index] = enc.transform(sub_df[col].values)
        return mof_ids

    def mof_to_ids(self, mof):
        ids = []
        for i, enc in enumerate(self.encoders.values()):
            arr = enc.transform([mof[i]])[0]
            ids.append(arr)
        return ids

    def ids_to_mof(self, ids):
        mof = []     
        for i, enc in enumerate(self.encoders.values()):
            cat = enc.inverse_transform([ids[i]])[0]
            mof.append(cat)

        return mof

    def ids_to_mof_(self, ids):
        mofs = []
        for index, id in enumerate(ids):
            mof = []     
            for i, enc in enumerate(self.encoders.values()):
                try:
                    cat = enc.inverse_transform([id[i]])[0]
                    mof.append(cat)
                except ValueError:
                    pass
            mofs.append(mof)

        return mofs

    def ids_array_to_mof_list(self, ids_arr):
        cats = []
        for i, enc in enumerate(self.encoders.values()):
            cats.append(enc.inverse_transform(ids_arr[:, i]))
        return np.array(cats).T.tolist()
