import os
import sys
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from astro_dataset import get_astro_data

sys.path.append('/n/home05/victoriaono/CAMELS/')
sys.path.append('/n/home05/victoriaono/CAMELS/model/utils')
from model import vdm_model, networks
from utils import draw_figure

torch.set_float32_matmul_precision("medium")


def train(
    model,
    datamodule,
):

    ckpt_path = None

    comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name="debiasing",
            experiment_name="simba-11.5-with-attn"
        )

    # Checkpoint every time val_loss improves
    val_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
    )

    # Checkpoint at every 3 epochs
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
        every_n_train_steps=3000,
        save_top_k = -1
    )

    trainer = Trainer(
        logger=comet_logger,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=1_000,
        gradient_clip_val=0.5,
        callbacks=[LearningRateMonitor(),
                    latest_checkpoint,
                    val_checkpoint]
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(7)
    cropsize = 256
    batch_size = 12
    num_workers = 20
    gamma_min = -13.3
    gamma_max = 13.3
    dataset = 'simba'
    embedding_dim = 48
    norm_groups = 8
    n_blocks = 4
    learning_rate = 1e-4

    dm = get_astro_data(
        num_workers=num_workers,
        # resize=cropsize,
        batch_size=batch_size,
        dataset=dataset
    )
    vdm = vdm_model.LightVDM(
        score_model=networks.UNetVDM(
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            embedding_dim=embedding_dim,
            norm_groups=norm_groups,
            n_blocks=n_blocks,
            # add_attention=True
        ),
        dataset=dataset,
        learning_rate=learning_rate,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        image_shape=(1,cropsize,cropsize),
        noise_schedule="learned_linear",
        draw_figure=draw_figure,
    )
    train(model=vdm, datamodule=dm)