import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings('ignore')

import torch_pruning as tp

import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainSegmentationDataset as Dataset
from logger import Logger
from loss import DiceLoss
from transform import transforms
from unet import UNet
from utils import log_images, dsc


def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    image_size = args.image_size
    example_inputs = torch.randn(1, 3, image_size, image_size)

    # 0. importance criterion for parameter selections
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')
    
    # 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
    ignored_layers = []
    # for m in model.modules():
    #     if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
    #         ignored_layers.append(m) # DO NOT prune the final classifier!


    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)

    unet.to(device)

    # 2. Pruner initialization
    prune_iters = args.prune_iters # You can prune your model to the target sparsity iteratively.
    pruner = tp.pruner.MagnitudePruner(
        unet, 
        example_inputs, 
        global_pruning=False, # If False, a uniform sparsity will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        iterative_steps=prune_iters, # the number of iterations to achieve target sparsity
        ch_sparsity=args.channel_sparsity, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    logger = Logger(args.logs)
    loss_train = []
    loss_valid = []

    step = 0

    base_macs, base_nparams = tp.utils.count_ops_and_params(unet, example_inputs)
    for i in range(prune_iters):
        # 3. the pruner.step will remove some channels from the unet with least importance
        pruner.step()
        
        # 4. Do whatever you like here, such as fintuning
        macs, nparams = tp.utils.count_ops_and_params(unet, example_inputs)
        # print(unet)
        print(unet(example_inputs).shape)
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i+1, prune_iters, base_nparams / 1e6, nparams / 1e6)
        )
        print(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i+1, prune_iters, base_macs / 1e9, macs / 1e9)
        )


        for epoch in tqdm(range(args.epochs), total=args.epochs):
            for phase in ["train", "valid"]:
                if phase == "train":
                    unet.train()
                else:
                    unet.eval()

                validation_pred = []
                validation_true = []

                for i, data in enumerate(loaders[phase]):
                    if phase == "train":
                        step += 1

                    x, y_true = data
                    x, y_true = x.to(device), y_true.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        y_pred = unet(x)

                        loss = dsc_loss(y_pred, y_true)

                        if phase == "valid":
                            loss_valid.append(loss.item())
                            y_pred_np = y_pred.detach().cpu().numpy()
                            validation_pred.extend(
                                [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                            )
                            y_true_np = y_true.detach().cpu().numpy()
                            validation_true.extend(
                                [y_true_np[s] for s in range(y_true_np.shape[0])]
                            )
                            if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                                if i * args.batch_size < args.vis_images:
                                    tag = "image/{}".format(i)
                                    num_images = args.vis_images - i * args.batch_size
                                    logger.image_list_summary(
                                        tag,
                                        log_images(x, y_true, y_pred)[:num_images],
                                        step,
                                    )

                        if phase == "train":
                            loss_train.append(loss.item())
                            loss.backward()
                            optimizer.step()

                    if phase == "train" and (step + 1) % 10 == 0:
                        log_loss_summary(logger, loss_train, step)
                        loss_train = []

                if phase == "valid":
                    log_loss_summary(logger, loss_valid, step, prefix="val_")
                    if sum([len(validation_pred[i]) for i in range(len(validation_pred))]) > 0:
                        mean_dsc = np.mean(dsc(
                            validation_pred,
                            validation_true,
                        ))
                    else:
                        mean_dsc = 0
                    logger.scalar_summary("val_dsc", mean_dsc, step)
                    if mean_dsc > best_validation_dsc:
                        best_validation_dsc = mean_dsc
                        torch.save(
                            unet.state_dict(),
                            os.path.join(args.weights, f'unet-{best_validation_dsc}.pt'))
                    loss_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        images_dir=args.images,
        subset="train",
        image_size=args.image_size,
        transform=transforms(scale=args.aug_scale,
                             angle=args.aug_angle,
                             flip_prob=0.5),
    )
    valid = Dataset(
        images_dir=args.images,
        subset="val",
        image_size=args.image_size,
        random_sampling=False,
    )
    return train, valid



def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument("--weights",
                        type=str,
                        default="./weights-binarizer",
                        help="folder to save weights")
    parser.add_argument("--logs",
                        type=str,
                        default="./logs-binarizer",
                        help="folder to save logs")
    parser.add_argument("--images",
                        type=str,
                        default="./data",
                        help="root folder with images")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )

    parser.add_argument(
        "--prune-iters",
        type=int,
        default=5,
        help="number of iterations for prunning",
    )

    parser.add_argument(
        "--channel-sparsity",
        type=float,
        default=0.7,
        help="remove 50\% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}",
    )

    args = parser.parse_args()
    main(args)
