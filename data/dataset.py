import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, args, flag='train'):

        data = np.load(args.data_path + '/' + args.dataset + '/' + args.dataset + '_train_data.npy')
        scaler = StandardScaler()
        scaler.fit(data[:int(0.7 * len(data)), :])

        if flag == 'train':
            data = np.load(args.data_path + '/' + args.dataset + '/' + args.dataset + '_train_data.npy')
            data = data[:int(0.7 * len(data)), :]
            label = np.zeros(len(data))
        elif flag == 'valid':
            data = np.load(args.data_path + '/' + args.dataset + '/' + args.dataset + '_train_data.npy')
            data = data[int(0.7 * len(data)):, :]
            label = np.zeros(len(data))
        else:
            data = np.load(args.data_path + '/' + args.dataset + '/' + args.dataset + '_test_data.npy')
            label = np.load(args.data_path + '/' + args.dataset + '/' + args.dataset + '_test_label.npy')

        data = pd.DataFrame(scaler.transform(data)).fillna(0).values

        self.data = data
        self.label = label
        self.window_size = args.window_size

    def __getitem__(self, index):
        x = self.data[index: index + self.window_size, :]
        y = self.label[index: index + self.window_size]

        return x, y

    def __len__(self):

        return len(self.data) - self.window_size + 1
