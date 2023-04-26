import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from models.InfoCTM import InfoCTM


class Runner:
    def __init__(self, args, params_list):
        self.args = args
        self.model = InfoCTM(args, *params_list)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def make_optimizer(self):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.args.learning_rate
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.args.lr_scheduler == 'StepLR':
            lr_scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_gamma, verbose=False)
        else:
            raise NotImplementedError(self.args.lr_scheduler)

        return lr_scheduler

    def train(self, data_loader):

        data_size = len(data_loader.dataset)
        num_batch = len(data_loader)
        optimizer = self.make_optimizer()

        if 'lr_scheduler' in self.args:
            lr_scheduler = self.make_lr_scheduler(optimizer)

        for epoch in range(1, self.args.epochs + 1):

            sum_loss = 0.

            loss_rst_dict = defaultdict(float)

            self.model.train()
            for batch_data in data_loader:
                batch_bow_en = batch_data['bow_en']
                batch_bow_cn = batch_data['bow_cn']
                params_list = [batch_bow_en, batch_bow_cn]

                rst_dict = self.model(*params_list)

                batch_loss = rst_dict['loss']

                for key in rst_dict:
                    if 'loss' in key:
                        loss_rst_dict[key] += rst_dict[key]

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                sum_loss += batch_loss.item() * len(batch_bow_en)

            if 'lr_scheduler' in self.args:
                lr_scheduler.step()

            sum_loss /= data_size

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_rst_dict:
                output_log += f' {key}: {loss_rst_dict[key] / num_batch :.3f}'

            print(output_log)

        beta_en, beta_cn = self.model.get_beta()
        beta_en = beta_en.detach().cpu().numpy()
        beta_cn = beta_cn.detach().cpu().numpy()
        return beta_en, beta_cn

    def get_theta(self, bow, lang):
        theta_list = list()
        data_size = bow.shape[0]
        all_idx = torch.split(torch.arange(data_size,), self.args.batch_size)
        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_bow = bow[idx]
                theta = self.model.get_theta(batch_bow, lang)
                theta_list.extend(theta.detach().cpu().numpy().tolist())

        return np.asarray(theta_list)

    def test(self, dataset):
        theta_en = self.get_theta(dataset.bow_en, lang='en')
        theta_cn = self.get_theta(dataset.bow_cn, lang='cn')
        return theta_en, theta_cn
