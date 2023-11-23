import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from lightning.pytorch import LightningModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convolutional block
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, 
                               kernel_size=3, padding=1, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, 
                               kernel_size=3, padding=1, padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
    
# Encoder block
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    # x: output of the conv_block and acts as the input of the pooling layer 
    # and as the skip connection feature map for the decoder block
    # p: output of the pooling layer
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    

# Decoder block
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv_up = nn.Conv2d(in_c, out_c, 
                                 kernel_size=3, padding=1, padding_mode='circular')
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = F.interpolate(inputs, scale_factor=2, mode='bilinear')
        x = self.conv_up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        
        return x

# Model
class LightUnet(LightningModule):
    def __init__(self,
                 learning_rate: float = 1.0e-3,
                 weight_decay: float = 1.0e-5):
        super().__init__()
        self.save_hyperparameters()

        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Image Output """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding_mode='circular')

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Image Output """
        outputs = self.outputs(d4)

        return outputs
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        if self.logger is not None:
            self.logger.log_metrics({"train_loss": loss}, step=batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        if self.logger is not None:
            self.logger.log_metrics({"val_loss": loss}, step=batch_idx)

        # generate maps and log on comet
        if batch_idx == 0:
            fig, ax = plt.subplots(ncols=3)            
            ax[0].imshow(x[0].squeeze().cpu(), cmap='cividis')
            ax[1].imshow(y[0].squeeze().cpu(), cmap='cividis')
            ax[2].imshow(y_pred[0].squeeze().cpu(), cmap='cividis')
            ax[0].set_title("Stars")
            ax[1].set_title("True DM")
            ax[2].set_title("Sampled DM")
            if self.logger is not None:
                self.logger.experiment.log_figure(figure=plt, step=self.current_epoch)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=10,

        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}