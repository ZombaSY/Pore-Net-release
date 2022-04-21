import torch
import time
import numpy as np
import os
import cv2

from models import utils
from models import model as model_hub
from models.dataloader import SegLoader
from PIL import Image


class Inferencer:

    def __init__(self, args):
        self.start_time = time.time()
        self.args = args

        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.loader_form = self.__init_data_loader(self.args.val_x_path,
                                                   self.args.val_y_path,
                                                   batch_size=1,
                                                   is_val=True)

        self.loader_val = self.loader_form.Loader

        self.model = self.__init_model()
        self.model.load_state_dict(torch.load(args.model_path))
        self.model.eval()

        if hasattr(self.args, 'mask_path'):
            self.mask_path = os.listdir(self.args.mask_path)

        self.data_length = len(os.listdir(self.args.val_x_path))

        dir_path, fn = os.path.split(self.args.model_path)
        fn, ext = os.path.splitext(fn)
        save_dir = dir_path + '/' + fn + '/'

        self.dir_path = dir_path
        self.model_fn = fn
        self.img_save_dir = save_dir
        self.num_batches_val = int(len(self.loader_val))
        self.metric = utils.StreamSegMetrics(2)
        self.threshold = float(self.args.threshold) * 255

        self.image_mean = self.loader_form.image_loader.image_mean
        self.image_std = self.loader_form.image_loader.image_std

    def start_inference_with_mask(self):
        self.metric.reset()
        tt_end = 0
        for batch_idx, (img, target) in enumerate(self.loader_val):
            with torch.no_grad():
                img, img_id = img
                target, _ = target

                path, fn = os.path.split(img_id[0])
                img_id, ext = os.path.splitext(fn)

                dir_path, fn = os.path.split(self.args.model_path)
                fn, ext = os.path.splitext(fn)
                save_dir = dir_path + '/' + fn + '/'

                tt = time.time()
                output = self.model(img)[0, :].cpu().detach()
                tt_end += time.time() - tt

                target = torch.where(target > 0, torch.tensor(1).float(), torch.tensor(0).float()).to(self.device)

                output_np = output.numpy().squeeze(0) * 255

                # _, output_argmax = cv2.threshold(output_np.astype(np.uint8), -1, 1,  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                _, output_argmax_np = cv2.threshold(output_np.astype(np.uint8), self.threshold, 1, cv2.THRESH_BINARY)
                output_argmax_np_reshaped = output_argmax_np.reshape((1, output_argmax_np.shape[0], output_np.shape[1]))

                img = img.squeeze(0).data.cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = img * np.array(self.image_std)
                img = img + np.array(self.image_mean)
                img = img * 255.0
                img = img.astype(np.uint8)

                # overlay
                output_argmax_np_reshaped = utils.label_img_to_color(torch.tensor(output_argmax_np_reshaped)).numpy().astype(np.uint8)

                mask = None
                for fn in self.mask_path:
                    mask_id, ext = os.path.splitext(fn)
                    if img_id in mask_id:
                        mask = fn

                if mask is None:
                    raise Exception('Cannot find txt file matching to image_id')

                image_mask = Image.open(self.args.mask_path + mask).convert('L')
                image_mask = image_mask.resize((self.args.input_size[1], self.args.input_size[0]))
                image_mask_np_3ch = np.array(image_mask)[:, :, None].repeat(3, axis=2)
                output_argmax_masked = np.bitwise_and(output_argmax_np_reshaped, image_mask_np_3ch)
                output_argmax_masked_map = 255 - output_argmax_masked

                img_overlay = np.bitwise_and(img, output_argmax_masked_map)
                img_overlay = img_overlay.astype(np.uint8)

                # heatmap
                output_heatmap = (output.squeeze(0).cpu().detach().numpy() * 255).astype(np.uint8)
                output_heatmap = 255 - output_heatmap
                output_heatmap = utils.grey_to_heatmap(output_heatmap)

                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                target_np = target.cpu().detach().squeeze(0).numpy().astype(np.uint8)
                output_argmax_np_tmp = output_argmax_np

                target_np_masked = np.bitwise_and(target_np, image_mask)
                output_argmax_np_tmp_masked = np.bitwise_and(output_argmax_np_tmp, image_mask)

                self.metric.update(target_np_masked, output_argmax_np_tmp_masked)

                Image.fromarray(img).save(save_dir + img_id + '.png', quality=100)
                Image.fromarray(output_argmax_np * 255).save(save_dir + img_id + '_argmax.png', quality=100)
                Image.fromarray(output_argmax_masked).save(save_dir + img_id + '_argmax_mask.png', quality=100)
                Image.fromarray(output_heatmap).save(save_dir + img_id + '_heatmap.png', quality=100)
                Image.fromarray(img_overlay).save(save_dir + img_id + '_overlay.png', quality=100)

                print(img_id + '\tDone!')

        metrics = self.metric.get_results()
        print(metrics)

    def __init_model(self):
        if self.args.model_name == 'Pore_Net':
            model = torch.nn.DataParallel(
                model_hub.PoreNet_SC().to(self.device))
        elif self.args.model_name == 'PoreNet_fSC':
            model = torch.nn.DataParallel(
                model_hub.PoreNet_fullySC().to(self.device))
        else:
            raise Exception('No model named', self.args.model_name)

        return model

    def __init_data_loader(self,
                           dataset_path,
                           label_path,
                           batch_size,
                           is_val=False):

        loader = SegLoader(dataset_path=dataset_path,
                           label_path=label_path,
                           batch_size=batch_size,
                           num_workers=self.args.worker,
                           pin_memory=self.args.pin_memory,
                           is_val=is_val,
                           args=self.args)

        return loader
