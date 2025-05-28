import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 

from utils.dataset_utils import DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR

import lightning.pytorch as pl
import torch.nn.functional as F
from torch.cuda.amp import autocast

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]

def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=4)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            
            with autocast():
                restored = net(degrad_patch)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)

                save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)

    parser.add_argument('--derain_path', type=str, default="test/derain/", help='save path of test raining images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="model.ckpt", help='checkpoint save path')
    testopt = parser.parse_args()
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path =  testopt.ckpt_name

    print("CKPT name : {}".format(ckpt_path))

    net  = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()


    print('Start testing rain streak removal...')
    derain_base_path = testopt.derain_path
    # for name in derain_splits:
    print('Start testing {} rain streak removal...'.format("noised"))
    # testopt.derain_path = os.path.join(derain_base_path,name)
    derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
    test_Derain_Dehaze(net, derain_set, task="derain")