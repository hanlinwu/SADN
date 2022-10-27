import argparse
import os

import numpy as np
import torch
from litsr.data import DownsampledDataset
from litsr.models import load_model
from litsr.utils import mkdir, read_yaml
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from tqdm import tqdm

from archs import *

seed_everything(123)


def test_pipeline(args):
    # setup scales and datasets
    test_datasets = (
        [_ for _ in args.datasets.split(",")] if args.datasets else ["sr-geo-15"]
    )

    # config ckpt path
    exp_path = os.path.dirname(os.path.dirname(args.checkpoint))
    ckpt_path = args.checkpoint

    # read config
    config = read_yaml(os.path.join(exp_path, "hparams.yaml"))

    # create model
    model = load_model(config, ckpt_path, strict=False)
    model.eval()

    # set gpu
    if args.gpus:
        model.cuda()

    scales = args.scales.split(",") if args.scales else [2, 3, 4]
    scales = [float(s) for s in scales]

    for dataset_name in test_datasets:
        for scale in scales:
            # config result path
            if args.self_ensemble:
                rslt_folder_name = "results_plus"
            else:
                rslt_folder_name = "results"
            rslt_path = os.path.join(
                exp_path, rslt_folder_name, dataset_name, "x" + str(scale)
            )

            mkdir(rslt_path)

            print(
                "==== Dataset {}, Scale Factor x{:.2f} ====".format(dataset_name, scale)
            )

            dataset = DownsampledDataset(
                datapath="load/benchmark/{0}/HR".format(dataset_name),
                scale=scale,
                is_train=False,
                cache="bin",
                rgb_range=config.data_module.args.rgb_range,
                mean=config.data_module.args.get("mean"),
                std=config.data_module.args.get("std"),
                return_img_name=True,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False
            )

            psnrs, ssims, run_times, losses = [], [], [], []
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataset)):
                if args.gpus:
                    lr, hr, name = batch
                    batch = (lr.cuda(), hr.cuda(), name)
                with torch.no_grad():
                    rslt = model.test_step(
                        batch,
                        batch_idx,
                        test_Y=False,
                        no_crop_border=False,
                        self_ensemble=args.self_ensemble,
                    )

                file_path = os.path.join(rslt_path, rslt["name"])

                if "log_img" in rslt.keys():
                    plt.imsave(file_path, rslt["log_img"])
                if "lr" in rslt.keys():
                    plt.imsave(file_path.replace(".png", "_lr.png"), rslt["lr"])
                if "val_loss" in rslt.keys():
                    losses.append(rslt["val_loss"])
                if "val_psnr" in rslt.keys():
                    psnrs.append(rslt["val_psnr"])
                if "val_ssim" in rslt.keys():
                    ssims.append(rslt["val_ssim"])
                if "time" in rslt.keys():
                    run_times.append(rslt["time"])

            if losses:
                mean_loss = np.array(losses).mean()
                print("- Loss: {:.4f}".format(mean_loss))
            if psnrs:
                mean_psnr = np.array(psnrs).mean()
                print("- PSNR: {:.4f}".format(mean_psnr))
            if ssims:
                mean_ssim = np.array(ssims).mean()
                print("- SSIM: {:.4f}".format(mean_ssim))
            if run_times:
                mean_runtime = np.array(run_times[1:]).mean()
                print("- Runtime : {:.4f}".format(mean_runtime))
            print("=" * 42)


def getTestParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="checkpoint index")
    parser.add_argument(
        "-g",
        "--gpus",
        default="0",
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    parser.add_argument("--datasets", default="", type=str, help="dataset names")
    parser.add_argument("--scales", default="", type=str, help="scale factors")
    parser.add_argument("--self_ensemble", action="store_true", help="self_ensemble")

    return parser


test_parser = getTestParser()

if __name__ == "__main__":
    args = test_parser.parse_args()
    test_pipeline(args)
