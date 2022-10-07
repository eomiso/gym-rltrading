import torch
import torch.nn as nn


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def float_tensor(arr, device='cpu'):
    if device == 'cuda':
        return torch.cuda.FloatTensor(arr)
    else:
        return torch.FloatTensor(arr)


def init_params(model, gain=1.0):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            m.bias.data.zero_()

        elif hasattr(m, '_modules'):
            for module in m._modules:
                try:
                    init_params(module, gain=gain)
                except AttributeError:
                    continue
                except Exception as e:
                    raise e
