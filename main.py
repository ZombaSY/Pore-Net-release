import torch

from train import Trainer
from inference import Inferencer
from torch.cuda import is_available
from datetime import datetime

import argparse
import wandb
import yaml
import numpy
import os


torch.manual_seed(0)    # fix seed for test
numpy.random.seed(0)
torch.backends.cudnn.benchmark = True


def conf_to_args(args, **kwargs):    # pass in variable numbers of args
    var = vars(args)

    for key, value in kwargs.items():
        var[key] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    arg = parser.parse_args()

    with open(arg.config_path, 'rb') as f:
        conf = yaml.load(f.read(), Loader=yaml.Loader)  # load the config file

    args = argparse.Namespace()
    conf_to_args(args, **conf)  # pass in your keyword args
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    now = datetime.now().strftime("%Y-%m-%d %H%M%S")

    if args.wandb:
        wandb.init(project='Pore-Net', config=args, name=now,
                   settings=wandb.Settings(start_method="fork"))

    print('Use CUDA :', args.cuda and is_available())

    if args.mode in ('train', 'calibrate'):
        trainer = Trainer(args, now)
        trainer.start_train()

    elif args.mode in 'inference':
        inferencer = Inferencer(args)

        if args.inference_mode == 'mask':
            inferencer.start_inference_with_mask()
    else:
        print('No mode supported.')


if __name__ == "__main__":
    main()
