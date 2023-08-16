import os
from time import time

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import MyDataset
from model.model import DCdetector
from utils.early_stopping import EarlyStopping
from utils.metric import getAffiliationMetrics


class Exp:
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting

        self.device = self._acquire_device()
        self.model_path, self.log_path, self.result_path = self._make_dirs()
        self.train_loader, self.valid_loader, self.test_loader = self._get_loader()

        self.model = DCdetector(self.args).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        self.early_stopping = EarlyStopping(patience=self.args.patience, path=self.model_path)
        self.discrepancy = nn.KLDivLoss()

        self.writer = SummaryWriter(log_dir=self.log_path)
        dummy_input = torch.randn(self.args.batch_size, self.args.window_size, self.args.channel)
        self.writer.add_graph(self.model, [dummy_input.float().to(self.device)])

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.devices)
            device = torch.device('cuda:{}'.format(self.args.devices))
            print('Use GPU: cuda:{}'.format(self.args.devices))
        else:
            device = torch.device('cpu')
            print('Use CPU')

        return device

    def _get_loader(self):
        train_set = MyDataset(self.args, flag='train')
        valid_set = MyDataset(self.args, flag='valid')
        test_set = MyDataset(self.args, flag='test')

        print('Train Data Shape: ', train_set.data.shape)
        print('Valid Data Shape: ', valid_set.data.shape)
        print('Test Data Shape: ', test_set.data.shape)
        print('Test Label Shape: ', test_set.label.shape)

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        valid_loader = DataLoader(valid_set, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False, drop_last=False)

        return train_loader, valid_loader, test_loader

    def _make_dirs(self):
        model_path = os.path.join(self.args.save_path + '/model/', self.setting)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        log_path = os.path.join(self.args.save_path + '/log/', self.setting)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        result_path = os.path.join(self.args.save_path + '/result/', self.setting)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        return model_path, log_path, result_path

    def _loss_fn(self, pos, neg):
        loss = 0.5 * self.discrepancy(pos.log(), neg.detach()) + 0.5 * self.discrepancy(neg.log(), pos.detach())

        return loss

    def _score_fn(self, pos, neg):
        score = []
        for i in range(len(pos)):
            score.append(0.5 * self.discrepancy(pos.log(), neg) + 0.5 * self.discrepancy(neg.log(), pos))
        score = torch.Tensor(score)

        return score

    def train(self):
        for e in range(self.args.epoch):
            start = time()

            self.model.train()
            train_loss = []
            for (batch_x, _) in tqdm(self.train_loader):
                self.optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                pos, neg = self.model(batch_x)
                loss = self._loss_fn(pos, neg)
                train_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                valid_loss = []
                for (batch_x, _) in tqdm(self.valid_loader):
                    batch_x = batch_x.float().to(self.device)
                    pos, neg = self.model(batch_x)
                    loss = self._loss_fn(pos, neg)
                    valid_loss.append(loss.item())

            train_loss, valid_loss = np.mean(train_loss), np.mean(valid_loss)

            end = time()

            print("lr = {:.10f}".format(self.optimizer.param_groups[0]['lr']))
            print("Epoch: {0} || Train Loss: {1:.6f} Valid Loss: {2:.6f} || Cost: {3:.6f}".format(
                e, train_loss, valid_loss, end - start)
            )

            self.writer.add_scalar('train/train_loss', train_loss, e)
            self.writer.add_scalar('train/valid_loss', valid_loss, e)

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break

        self.writer.close()

    def test(self):
        self.model.load_state_dict(torch.load(self.model_path + '/' + 'checkpoint.pth'))

        with torch.no_grad():
            self.model.eval()

            valid_score = []
            for (batch_x, _) in tqdm(self.valid_loader):
                batch_x = batch_x.float().to(self.device)
                pos, neg = self.model(batch_x)
                score = self._score_fn(pos, neg)
                valid_score.append(score.detach().cpu().numpy())

            test_score, test_label = [], []
            for (batch_x, batch_label) in tqdm(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_label = batch_label.float().to(self.device)
                pos, neg = self.model(batch_x)
                score = self._score_fn(pos, neg)
                test_score.append(score.detach().cpu().numpy())
                test_label.append(batch_label.detach().cpu().numpy())

        valid_score = np.concatenate(valid_score, axis=0)
        test_score = np.concatenate(test_score, axis=0)
        test_label = np.concatenate(test_label, axis=0)[:, -1]

        threshold = np.percentile(valid_score, 100 - self.args.anomaly_ratio)

        print('Valid Score Shape:', valid_score.shape)
        print('Test Score Shape:', test_score.shape)
        print('Test Label Shape:', test_label.shape)
        print('Threshold: ', threshold)

        np.save(self.result_path + '/' + 'valid_score', valid_score)
        np.save(self.result_path + '/' + 'test_score', test_score)
        np.save(self.result_path + '/' + 'test_label', test_label)

        test_pred = np.array(test_score > threshold).astype(int)
        p, r, f1 = getAffiliationMetrics(test_label, test_pred)

        print('P: {0:.4f}, R: {1:.4f}, F1: {2:.4f}'.format(p, r, f1))

        f = open("result.txt", 'a')
        f.write(self.setting + "  \n")
        f.write('P: {0:.4f}, R: {1:.4f}, F1: {2:.4f}'.format(p, r, f1))
        f.write('\n')
        f.write('\n')
        f.close()
