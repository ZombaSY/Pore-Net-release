import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
import os

from models import model as model_hub
from models import loss as loss_hub
from models import utils
from torchsummary import summary


def initialize_weights(layer, activation='relu'):

    for module in layer.modules():
        module_name = module.__class__.__name__

        if activation in ('relu', 'leaky_relu'):
            layer_init_func = nn.init.kaiming_uniform_
        elif activation == 'tanh':
            layer_init_func = nn.init.xavier_uniform_
        else:
            raise Exception('Please specify your activation function name')

        if hasattr(module, 'weight'):
            if module_name.find('Conv2') != -1:
                layer_init_func(module.weight)
            elif module_name.find('BatchNorm') != -1:
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)
            elif module_name.find('Linear') != -1:
                layer_init_func(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.1)
            else:
                # print('Cannot initialize the layer :', module_name)
                pass
        else:
            pass


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    x = torch.rand(1, 3, 1280, 1024).cuda()
    # x2 = F.one_hot(torch.arange(0, 2), num_classes=36).cuda().float()
    # x = (x, x2)
    y = torch.rand(1, 1, 1280, 1024)
    y = torch.where(y >= 0.5, torch.tensor(1), torch.tensor(0)).cuda()

    model = model_hub.PoreNet_SC_H2().cuda()

    loss_criterion = loss_hub.FocalTverskyLoss().cuda()
    # if self.args.criterion == 'MSE':
    #     loss = loss_hub.MSELoss().to(self.device)
    # elif self.args.criterion == 'BCE':
    #     loss = loss_hub.BCELoss().to(self.device)
    # elif self.args.criterion == 'Dice':
    #     loss = loss_hub.DiceLoss().to(self.device)
    # elif self.args.criterion == 'DiceBCE':
    #     loss = loss_hub.DiceBCELoss().to(self.device)
    # elif self.args.criterion == 'FocalBCE':
    #     loss = loss_hub.FocalBCELoss().to(self.device)
    # elif self.args.criterion == 'Tversky':
    #     loss = loss_hub.TverskyLoss().to(self.device)
    # elif self.args.criterion == 'FocalTversky':
    #     loss = loss_hub.FocalTverskyLoss().to(self.device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

    model.train()
    for i in range(50):
        optimizer.zero_grad()

        out = model(x)
        loss = loss_criterion(out, y)

        print(loss)
        # ---- backward ----
        loss.backward()
        optimizer.step()

    # m = torch.jit.script(model)
    # torch.jit.save(m, 'swin.pt')

    # torch.save(model.state_dict(), 'MobNet.pt')

    # from ptflops import get_model_complexity_info
    #
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (3, 1024, 1280), as_strings=True,
    #                                              print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #
    # summary(model.cuda(), (3, 1024, 1280))

    # model.eval()
    # t_sum = []
    # for i in range(11):
    #     optimizer.zero_grad()
    #
    #     tt = time.time()
    #     _ = model(x)
    #
    #     if i >= 1:  # ignore first batch
    #         t_sum.append(time.time() - tt)
    #         # print(t_sum)
    #
    # print('mean:', np.array(t_sum).mean())
    # print('max mem:', torch.cuda.max_memory_allocated())


if __name__ == '__main__':
    main()
