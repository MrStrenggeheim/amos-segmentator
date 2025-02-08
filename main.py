import argparse
import glob
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary
from torchvision.utils import make_grid
from utils.amos import AMOSDataModule


class SegmentationModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.lr = args.lr
        self.criterion = DiceLoss(
            include_background=True, to_onehot_y=True, softmax=True, reduction="mean"
        )

        if args.model_type == "pretrained":
            self.model = models.segmentation.fcn_resnet50(pretrained=True)
            self.model.classifier[4] = nn.Conv2d(512, 73, kernel_size=1)
        else:
            self.model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=73,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        # cosine scheduling, Tmax = number of epochs to reach min lr
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        return (
            self.model(x)["out"]
            if isinstance(self.model, models.segmentation.FCN)
            else self.model(x)
        )

    def training_step(self, batch, batch_idx):
        images, targets = (batch["images"], batch["images_target"])
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch["images"], batch["images_target"]
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)

        # also save images
        if batch_idx == 0:
            self.log("lr", self.lr_schedulers().get_last_lr()[0])
            exp = self.trainer.logger.experiment
            # convert onehot back to class idx
            outputs = torch.argmax(outputs, dim=1, keepdim=True).float()
            imgage_grid = self._make_grid(images)
            target_grid = self._make_grid(targets.float())
            output_grid = self._make_grid(outputs)
            exp.add_image("input", imgage_grid, self.global_step)
            exp.add_image("target", target_grid, self.global_step)
            exp.add_image("output", output_grid, self.global_step)
            return loss

    def _make_grid(self, x):
        return make_grid(x[:16], normalize=True, scale_each=False, nrow=4)


def get_last_checkpoint(log_dir):
    ckpts = sorted(
        glob.glob(os.path.join(log_dir, "**/checkpoints/*.ckpt"), recursive=True)
    )
    return ckpts[-1] if ckpts else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="amos_segmentator")
    parser.add_argument("--log_dir", type=str, default="tb_logs")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--seg_dir", type=str, required=True)
    parser.add_argument("--transforms", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument(
        "--model_type", type=str, choices=["unet", "pretrained"], default="unet"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # tech setup
    pl.seed_everything(0)
    torch.set_float32_matmul_precision("high")  # allow tensor cores

    dataset_module = AMOSDataModule(
        args.img_dir,
        args.seg_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        transforms=args.transforms,
        num_workers=args.num_workers,
    )

    logger = TensorBoardLogger(args.log_dir, name=args.exp_name)
    logger.log_hyperparams(vars(args))

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="amos_segmentator_checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        save_last=True,
    )

    model = SegmentationModel(args)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=50,
        logger=logger,
        log_every_n_steps=25,
        val_check_interval=0.25,
        limit_val_batches=0.25,
        callbacks=[checkpoint_callback],
        fast_dev_run=False,
    )

    summary(model, (1, args.img_size, args.img_size), depth=10)

    last_ckpt = None
    if args.resume:
        last_ckpt = get_last_checkpoint(os.path.join(args.log_dir, args.exp_name))
        if last_ckpt:
            print(f"Resuming from checkpoint: {last_ckpt}")
        else:
            print("No checkpoint found. Training from scratch.")
    trainer.fit(model, dataset_module, ckpt_path=last_ckpt)


# archive #####################
# torch.save(model.state_dict(), "amos_segmentator.pth")
# self.entropy_loss = nn.CrossEntropyLoss(
#     # wight bg less
#     # weight=torch.from_numpy(np.array([0.1] + [1.0] * 72)).float()
# )
# targets_onehot = (
#     torch.nn.functional.one_hot(targets.long().squeeze(), num_classes=73)
#     .permute(0, 3, 1, 2)
#     .float()
# )
