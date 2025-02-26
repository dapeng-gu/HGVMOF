import os, random, gc
import pickle


class DataFolder(object):
    def __init__(self, data_folder, batch_size, shuffle=True):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data_files) * 1000

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                batches = pickle.load(f)
            if self.shuffle: random.shuffle(batches)  #shuffle data before batch
            for batch in batches:
                yield batch
            del batches
            gc.collect()
