import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split, TensorDataset, DataLoader

from tqdm.notebook import tqdm, trange
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from translate import Translate
from augment import AugmentedDataset

import time
import pandas

np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load train data
mass_cdm = np.load('../data/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy')
mass_mstar = np.load('../data/Maps_Mstar_IllustrisTNG_LH_z=0.00.npy')
params = pandas.read_csv('../data/CMD_2D_maps_data_params_IllustrisTNG.txt', 
                         sep=' ', header=None, usecols=[0,1])
params = params.rename(columns={0:'Omega_m',1:'sigma_8'})
params = params.sort_values(by=['Omega_m', 'sigma_8'], ascending=False)

# Normalize dataset
mass_mstar = np.log10(mass_mstar+1)
mass_cdm = np.log10(mass_cdm)
mass_mstar = (mass_mstar - mass_mstar.mean()) / mass_mstar.std()
mass_cdm = (mass_cdm - mass_cdm.mean()) / mass_cdm.std()

cdm_min = mass_cdm[0].min()
cdm_max = mass_cdm[0].max()
mstar_min = mass_mstar[0].min()
mstar_max = mass_mstar[0].max()

# Split data into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(mass_mstar, mass_cdm, test_size=0.2)

# Transform to torch tensor, adding a channel dimension
x_train_tensor = torch.Tensor(x_train).unsqueeze(1)
y_train_tensor = torch.Tensor(y_train).unsqueeze(1)

x_val_tensor = torch.Tensor(x_val).unsqueeze(1)
y_val_tensor = torch.Tensor(y_val).unsqueeze(1)

# Create datasets
# Does this mean all the datapoints are augmented? Or are datapoints sometimes original?
train_dataset = AugmentedDataset(x_train_tensor, 
                                 y_train_tensor, 
                                 transform=Translate())
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

# Create train and validation data loaders
BATCH_SIZE = 64
train_iterator = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=BATCH_SIZE)
val_iterator = DataLoader(val_dataset,
                          batch_size=BATCH_SIZE)


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
    
    
# Build model
class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

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


model = build_unet()
model = model.to(device)

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

loss = nn.L1Loss()

history = {'train_loss': [], 'valid_loss': []}

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0

    model.train()

    for (x, y) in tqdm(iterator, desc='Training'):
        
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device, epoch):
    epoch_loss = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in tqdm(iterator, desc='Evaluating'):

            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            epoch_loss += loss.item()
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(y_pred[0].squeeze().cpu().numpy(), vmin=cdm_min, vmax=cdm_max)
    ax[1].imshow(y[0].squeeze().cpu().numpy(), vmin=cdm_min, vmax=cdm_max)
    plt.savefig(f'./images_test/pred_{epoch}.png', dpi=300)
    plt.close()
    
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Train model
plt.style.use('seaborn-v0_8-dark')
epochs = 10

best_valid_loss = 10

for epoch in trange(epochs, desc='Epochs'):

    start_time = time.monotonic()

    train_loss = train(model, train_iterator, optimizer, loss, device)
    valid_loss = evaluate(model, val_iterator, loss, device, epoch)
    
    scheduler.step(valid_loss)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss,
            'learning_rate': scheduler.state_dict()
            }, 'unet-model.pt')

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    history['train_loss'].append(train_loss)
    history['valid_loss'].append(valid_loss)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s', flush=True)
    print(f'\tTrain Loss: {train_loss:.3f}', flush=True)
    print(f'\t Val. Loss: {valid_loss:.3f}', flush=True)
    

# Plot the loss curve
plt.figure()
plt.plot(history['train_loss'], label='train_loss', color='#988ED5')
plt.plot(history['valid_loss'], label='valid_loss', color='#FFB5B8')
plt.title('Training Loss on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./images_test/train_loss_curve.png', dpi=300)
plt.close()

checkpoint = torch.load('unet-model.pt')

# Get images with the highest Omega values
highest_indices = params[:10].index.values
sample_mstar = mass_mstar[highest_indices]
sample_cdm = mass_cdm[highest_indices]

x_sample_tensor = torch.Tensor(sample_mstar).unsqueeze(1)
y_sample_tensor = torch.Tensor(sample_cdm).unsqueeze(1)

sample_dataset = TensorDataset(x_sample_tensor,y_sample_tensor)
sample_iterator = DataLoader(sample_dataset)

# Plot histograms
fig, axes = plt.subplots(5, 2, sharey=True, figsize=((6, 8)))

model = build_unet().to(device)
model.load_state_dict(checkpoint['model_state_dict'])

axs = axes.flatten()

y_preds = []

model.eval()
with torch.no_grad():
    for i, (x, y) in enumerate(sample_iterator):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        y_preds.append(y_pred.squeeze().cpu().numpy())
        x_range = np.linspace(-2, 6, 30)
        axs[i].hist(y_pred.squeeze().cpu().numpy().flatten(), 
                    alpha=0.8, 
                    density=True,
                    bins=x_range)
        axs[i].hist(mass_cdm[i].flatten(),
                    alpha=0.8, 
                    density=True,
                    bins=x_range)
fig.suptitle('True vs Predicted CDM density')
fig.tight_layout()
plt.savefig('./images_test/cdm_hist.png', dpi=300)

# do hexbin using seaborn, draw diagonal line
# to test, run it for 1 epoch and see if images work


# rotate image - mosaic images

# could also plot trained images