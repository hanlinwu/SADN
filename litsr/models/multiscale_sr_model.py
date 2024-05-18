import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from litsr.archs import create_net
from litsr.metrics import calc_psnr_ssim
from litsr.models.utils import forward_self_ensemble
from litsr.transforms import tensor2uint8
from litsr.utils.logger import logger
from litsr.utils.registry import ModelRegistry
from thop import profile


@ModelRegistry.register()
class MultiScaleSRModel(pl.LightningModule):
    """
    Basic Multi-scale SR Model optimized by pixel-wise loss
    """

    def __init__(self, opt):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(opt)
        self.opt = opt.lit_model.args

        # init super-resolution network
        self.sr_net = create_net(self.opt["network"])

        # define loss function (L1 loss)
        self.loss_fn = nn.L1Loss()

    def forward(self, lr, out_size):
        return self.sr_net(lr, out_size)

    def forward_self_ensemble(self, lr, out_size):
        return forward_self_ensemble(self, lr, out_size)

    def training_step(self, batch, *args):
        lr, hr = batch
        out_size = hr.shape[2:]
        sr = self.forward(lr, out_size)
        loss = self.loss_fn(sr, hr)
        return loss

    def training_epoch_end(self, outputs):
        if hasattr(self.sr_net, "update_temperature"):
            logger.warning("temperature updated!")
            self.sr_net.update_temperature()
        avg_loss = torch.stack([out["loss"] for out in outputs]).mean()
        self.log("train/loss", avg_loss, on_epoch=True)
        return

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        lr, hr, name = batch
        out_size = hr.shape[2:]
        torch.cuda.synchronize()
        start = time.time()
        if self.opt.valid.get("self_ensemble") or kwargs.get("self_ensemble"):
            sr = self.forward_self_ensemble(lr, out_size).detach()
        else:
            sr = self.forward(lr, out_size).detach()
        torch.cuda.synchronize()
        end = time.time()

        loss = self.loss_fn(sr, hr).item()

        sr_np, hr_np = tensor2uint8([sr.cpu()[0], hr.cpu()[0]], self.opt.rgb_range)

        if kwargs.get("no_crop_border", self.opt.valid.get("no_crop_border")):
            crop_border = 0
        else:
            crop_border = int(np.ceil(float(hr.shape[2]) / lr.shape[2]))

        test_Y = kwargs.get("test_Y", self.opt.valid.get("test_Y", True))

        if batch_idx == 0:
            logger.warning("Test_Y: {0}, crop_border: {1}".format(test_Y, crop_border))

        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=crop_border, test_Y=test_Y
        )
        return {
            "val_loss": loss,
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img": sr_np,
            "name": name[0],
            "time": end - start,
        }

    def validation_epoch_end(self, outputs):
        tensorboard = self.logger.experiment
        psnr_list = []
        if type(outputs[0]) == dict:
            outputs = [outputs]

        for idx, output in enumerate(outputs):
            scale = self.hparams.data_module.args.valid.scales[idx]
            tensorboard = self.logger.experiment

            log_img = output[0]["log_img"]

            avg_psnr = np.array([x["val_psnr"] for x in output]).mean()
            psnr_list.append(avg_psnr)

            self.log("val/psnr_x{0}".format(str(scale)), avg_psnr, on_epoch=True)
            tensorboard.add_image(
                "SR/{0}".format(str(scale)),
                log_img,
                self.global_step,
                dataformats="HWC",
            )
            psnr_list.append(avg_psnr)

        self.log(
            "val/psnr",
            np.array(psnr_list).mean(),
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

    def test_step(self, batch, batch_idx, *args, **kwargs):
        lr, hr, name = batch
        out_size = hr.shape[2:]
        torch.cuda.synchronize()
        start = time.time()
        if self.opt.valid.get("self_ensemble") or kwargs.get("self_ensemble"):
            sr = self.forward_self_ensemble(lr, out_size).detach()
        else:
            sr = self.forward(lr, out_size).detach()
        torch.cuda.synchronize()
        end = time.time()

        loss = self.loss_fn(sr, hr).item()

        sr_np, hr_np = tensor2uint8([sr.cpu()[0], hr.cpu()[0]], self.opt.rgb_range)

        if kwargs.get("no_crop_border", self.opt.valid.get("no_crop_border")):
            crop_border = 0
        else:
            crop_border = int(np.ceil(float(hr.shape[2]) / lr.shape[2]))

        test_Y = kwargs.get("test_Y", self.opt.valid.get("test_Y", True))

        if batch_idx == 0:
            logger.warning("Test_Y: {0}, crop_border: {1}".format(test_Y, crop_border))

        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=crop_border, test_Y=test_Y
        )
        flops, params = profile(
            self.sr_net,
            inputs=(lr, out_size),
            verbose=False,
            custom_ops={nn.ReLU: None},
        )
        return {
            "val_loss": loss,
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img": sr_np,
            "name": name[0],
            "time": end - start,
            "flops": flops,
            "params": params,
        }

    def configure_optimizers(self):
        betas = self.opt.optimizer.get("betas") or (0.9, 0.999)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.opt.optimizer.lr, betas=betas
        )
        if self.opt.optimizer.get("lr_scheduler_step"):
            LR_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.opt.optimizer.lr_scheduler_step,
                gamma=self.opt.optimizer.lr_scheduler_gamma,
            )
        elif self.opt.optimizer.get("lr_scheduler_milestones"):
            LR_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.opt.optimizer.lr_scheduler_milestones,
                gamma=self.opt.optimizer.lr_scheduler_gamma,
            )
        else:
            raise Exception("No lr settings found! ")
        return [optimizer], [LR_scheduler]
