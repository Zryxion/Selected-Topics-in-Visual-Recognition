import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        self.outputs = []  # Store (restored, ground_truth) for PSNR

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)

        self.outputs.append((restored.detach(), clean_patch.detach()))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        total_psnr = 0.0
        count = 0
        for restored, clean in self.outputs:
            for r, c in zip(restored, clean):
                mse = F.mse_loss(r, c)
                psnr = 10 * torch.log10(1.0 / mse)
                total_psnr += psnr.item()
                count += 1
        avg_psnr = total_psnr / max(count, 1)
        self.log("train_psnr", avg_psnr, prog_bar=True, sync_dist=True)
        self.outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=150
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)


def main():
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument(
        '--epochs', type=int, default=120,
        help='maximum number of epochs to train the total model.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help="Batch size to use per GPU"
    )
    parser.add_argument(
        '--lr', type=float, default=2e-4,
        help='learning rate of encoder.'
    )
    parser.add_argument(
        '--de_type', nargs='+',
        default=['derain', 'dehaze'],
        help='which type of degradations is training and testing for.'
    )
    parser.add_argument(
        '--patch_size', type=int, default=128,
        help='patchsize of input.'
    )
    parser.add_argument(
        '--num_workers', type=int, default=16,
        help='number of workers.'
    )

    # Paths
    parser.add_argument(
        '--data_file_dir', type=str, default='data_dir/',
        help='where clean images of denoising saves.'
    )
    parser.add_argument(
        '--derain_dir', type=str, default='data/Train/Derain/',
        help='where training images of deraining saves.'
    )
    parser.add_argument(
        '--dehaze_dir', type=str, default='data/Train/Dehaze/',
        help='where training images of dehazing saves.'
    )
    parser.add_argument(
        '--output_path', type=str, default="output/",
        help='output save path'
    )
    parser.add_argument(
        '--ckpt_path', type=str, default="ckpt/Denoise/",
        help='checkpoint save path'
    )
    parser.add_argument(
        "--wblogger", type=str, default="promptir",
        help="Determine to log to wandb or not and the project name"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="train_ckpt",
        help="Name of the Directory where the checkpoint is to be saved"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=4,
        help="Number of GPUs to use for training"
    )

    opt = parser.parse_args()

    print("Options")
    print(opt)

    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir, every_n_epochs=2, save_top_k=-1
    )
    trainloader = DataLoader(
        trainset, batch_size=opt.batch_size, pin_memory=True,
        shuffle=True, drop_last=True, num_workers=opt.num_workers
    )

    model = PromptIRModel()
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=[0, 1],
        precision="16-mixed",
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        # ckpt_path="train_ckpt/epoch=149-step=80484.ckpt"
    )


if __name__ == '__main__':
    main()
