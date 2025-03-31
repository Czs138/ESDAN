import torch
import torch.nn as nn


class smam_module(torch.nn.Module):
    def __init__(self, nf, e_lambda=1e-4, kernel_size = 3):
        super(smam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

        self.k1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        x = self.k1(x)
        return x * self.activaton(y)
