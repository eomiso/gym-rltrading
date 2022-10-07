from abc import ABC, abstractmethod
import os
from envs import trading
from algo.utils import float_tensor
import torch


class Algorithm(ABC):
    def __init__(self, args):
        self.env = trading(args)
        self.obs = float_tensor(self.env.reset(), device=args.device)
        self.args = args
        self.step = 0

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    def save_model(self, tag=None):
        if tag is None:
            filename = 'model.pt'
            path = os.path.join(self.logger.log_dir, filename)
        else:
            path = os.path.join(self.logger.log_dir, 'checkpoints')
            if not os.path.exists(path):
                os.mkdir(path)
            filename = tag + '.pt'
            path = os.path.join(path, filename)
        torch.save(self.model, path)

    def load_model(self, checkpoint):
        self.model = torch.load(checkpoint).to(self.args.device)

    def plot_actions(self):
        acs_hist = torch.cat(list(self.acs_buf))
        for i in range(acs_hist.size(1)):
            self.logger.add_histogram(
                'Actions/' + str(i),
                acs_hist[:, i],
                self.step
            )
