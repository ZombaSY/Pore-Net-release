import torch
import time
import os
import wandb

from models import model as model_hub
from models.dataloader import SegLoader
from datetime import datetime
from models import lr_scheduler
from models import utils
from models import loss as loss_hub


class Trainer:
    def __init__(self, args, now=None):

        self.start_time = time.time()
        self.args = args

        if not os.path.exists(self.args.saved_model_directory):
            os.mkdir(self.args.saved_model_directory)

        # Check cuda available and assign to device
        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # 'init' means that this variable must be initialized.
        # 'set' means that this variable is available of being set, not must.
        self.loader_train = self.__init_data_loader(self.args.train_x_path,
                                                    self.args.train_y_path,
                                                    self.args.batch_size,
                                                    is_val=False,
                                                    args=self.args,
                                                    # mask_path=self.args.train_mask_path
                                                    )
        self.loader_val = self.__init_data_loader(self.args.val_x_path,
                                                  self.args.val_y_path,
                                                  batch_size=1,
                                                  is_val=True,
                                                  args=self.args,
                                                  # mask_path=self.args.val_mask_path
                                                  )

        if self.args.mode == 'train' or self.args.mode == 'calibrate':
            self.model = self.__init_model()
            self.optimizer = self.__init_optimizer()
            self.scheduler = self.__set_scheduler()

        self.criterion = self.__init_criterion()

        if self.args.wandb:
            if self.args.mode == 'train':
                wandb.watch(self.model)

        now_time = now if now is not None else datetime.now().strftime("%Y%m%d %H%M%S")
        self.saved_model_directory = self.args.saved_model_directory + '/' + now_time
        self.num_batches_train = int(len(self.loader_train))
        self.num_batches_val = int(len(self.loader_val))

        self.metric = utils.StreamSegMetrics(2)
        self.metric_best = 0
        self.model_post_path = None

        if hasattr(self.args, 'mask_path'):
            self.mask_path = os.listdir(self.args.mask_path)

    def __train(self, epoch):
        self.model.train()
        loss_mean = 0

        print('Start Train')
        for batch_idx, (img, target) in enumerate(self.loader_train):
            if (img[0].shape[0] / torch.cuda.device_count()) <= 1:    # if has 1 batch per GPU
                break   # avoid BN issue

            img, _ = img
            target, roi_mask = target

            target = torch.where(target > 0, torch.tensor(1), torch.tensor(0)).to(self.device)

            output = self.model(img)
            output = output.squeeze(1)

            # apply ROI mask
            if hasattr(self.args, 'train_mask_path') and hasattr(self.args, 'val_mask_path'):
                roi_mask = roi_mask.cuda()
                output = output * roi_mask
                target = target * roi_mask

            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_mean += loss.item()

        self.scheduler.step()

        loss_mean /= self.num_batches_train
        print('{} epoch / Train Loss {} : {}'.format(epoch, self.args.criterion, loss_mean))

        if self.args.wandb:
            wandb.log({'Train Loss {}'.format(self.args.criterion): loss_mean})

    def __validate(self, epoch):
        self.model.eval()

        print('Start Validation')

        for batch_idx, (img, target) in enumerate(self.loader_val):
            with torch.no_grad():
                img, img_id = img
                target, roi_mask = target

                target = torch.where(target > 0, torch.tensor(1), torch.tensor(0)).to(self.device)

                output = self.model(img)
                output = output.squeeze(1)

                # apply ROI mask
                if hasattr(self.args, 'train_mask_path') and hasattr(self.args, 'val_mask_path'):
                    roi_mask = roi_mask.cuda()
                    output = output * roi_mask
                    target = target * roi_mask

                output_th_np = utils.threshold(output, self.args.threshold)
                self.metric.update(target.cpu().detach().numpy(), output_th_np)

        metrics = self.metric.get_results()
        iou_background = metrics['Class IoU'][0]
        iou_target = metrics['Class IoU'][1]
        mIoU = (iou_background + iou_target) / 2

        print('{} epoch / Val mIoU Score : {}'.format(epoch, mIoU))

        if self.args.wandb:
            wandb.log({'Val Background IoU': iou_background,
                       'Val Pore IoU': iou_target,
                       'Val mIoU Score': mIoU})

        # save best model
        if iou_target > self.metric_best:
            self.metric_best = iou_target
            self.save_model(self.args.model_name, epoch, iou_target, best_flag=True)

    def start_train(self):
        for epoch in range(1, self.args.epoch + 1):
            if self.args.mode == 'train' or self.args.mode == 'calibrate':
                self.metric.reset()
                self.__train(epoch)
                self.__validate(epoch)

            print('### {} / {} epoch ended###'.format(epoch, self.args.epoch))

    def save_model(self, model_name, epoch, metric=None, best_flag=False):
        file_path = self.saved_model_directory + '/'

        if best_flag:
            if self.model_post_path is not None:
                os.remove(self.model_post_path)

            file_format = file_path + model_name + '_Best_' + str(epoch) + '_metric_' + str(metric) + '.pt'
            self.model_post_path = file_format

            if not os.path.exists(file_path):
                os.mkdir(file_path)

            if self.args.mode == 'train' or self.args.mode == 'calibrate':
                torch.save(self.model.state_dict(), file_format)

        else:
            file_format = file_path + model_name + '_' + str(epoch) + '_metric_' + str(metric) + '.pt'

            if not os.path.exists(file_path):
                os.mkdir(file_path)

            if self.args.mode == 'train' or self.args.mode == 'calibrate':
                torch.save(self.model.state_dict(), file_format)

        print(file_format + '\t model saved!!')

    def __init_data_loader(self,
                           dataset_path,
                           label_path,
                           batch_size,
                           is_val=False,
                           mask_path=None,
                           **kwargs):

        if mask_path is None and mask_path is None:
            loader = SegLoader(dataset_path=dataset_path,
                               label_path=label_path,
                               batch_size=batch_size,
                               num_workers=self.args.worker,
                               pin_memory=self.args.pin_memory,
                               is_val=is_val,
                               **kwargs)
        else:
            loader = SegLoader(dataset_path=dataset_path,
                               label_path=label_path,
                               batch_size=batch_size,
                               num_workers=self.args.worker,
                               pin_memory=self.args.pin_memory,
                               is_val=is_val,
                               mask_path=mask_path,
                               **kwargs)

        return loader.Loader

    def __init_model(self):
        model = None

        if self.args.mode == 'train' or self.args.mode == 'calibrate':
            if self.args.model_name == 'Pore_Net':
                model = torch.nn.DataParallel(
                    model_hub.PoreNet_SC().to(self.device))
            elif self.args.model_name == 'PoreNet_fSC':
                model = torch.nn.DataParallel(
                    model_hub.PoreNet_fullySC().to(self.device))
            else:
                raise Exception('No model named', self.args.model_name)

        return model

    def __init_criterion(self):
        if self.args.mode == 'train' or self.args.mode == 'calibrate':
            if self.args.criterion == 'MSE':
                loss = loss_hub.MSELoss().to(self.device)
            elif self.args.criterion == 'BCE':
                loss = loss_hub.BCELoss().to(self.device)
            elif self.args.criterion == 'Dice':
                loss = loss_hub.DiceLoss().to(self.device)
            elif self.args.criterion == 'DiceBCE':
                loss = loss_hub.DiceBCELoss().to(self.device)
            elif self.args.criterion == 'FocalBCE':
                loss = loss_hub.FocalBCELoss().to(self.device)
            elif self.args.criterion == 'Tversky':
                loss = loss_hub.TverskyLoss().to(self.device)
            elif self.args.criterion == 'FocalTversky':
                loss = loss_hub.FocalTverskyLoss().to(self.device)
            else:
                raise Exception('No criterion named', self.args.model_name)

        return loss

    def __init_optimizer(self):
        optimizer = None

        if self.args.mode == 'train' or self.args.mode == 'calibrate':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

        return optimizer

    def __set_scheduler(self):
        scheduler = None

        if self.args.mode == 'train' or self.args.mode == 'calibrate':
            if hasattr(self.args, 'scheduler'):
                if self.args.scheduler == 'WarmupCosine':
                    scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=self.optimizer,
                                                                  warmup_steps=1,
                                                                  t_total=self.loader_train.__len__(),
                                                                  cycles=0.01,
                                                                  last_epoch=-1)
                elif self.args.scheduler == 'CosineAnnealingLR':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0.000001)

                else:
                    raise Exception('Scheduler initialization error!!!')
            else:
                scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=self.optimizer,
                                                              warmup_steps=1,
                                                              t_total=self.loader_train.__len__(),
                                                              cycles=0.01,
                                                              last_epoch=-1)

        return scheduler
