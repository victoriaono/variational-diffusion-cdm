import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from lightning.pytorch import LightningDataModule
import h5py

from .augmentation import Translate, Permutate, Flip, Normalize, Resize
from .constants import norms

class AstroDataset(TensorDataset):
    def __init__(self, m_star, m_cdm, transform=None):
        assert len(m_star) == len(m_cdm)
        self.m_star = m_star
        self.m_cdm = m_cdm
        self.transform = transform

    def __len__(self):
        return len(self.m_star)

    def __getitem__(self, index):
        m_star = self.m_star[index]
        m_cdm = self.m_cdm[index]

        if self.transform:
            m_star, m_cdm = self.transform((m_star, m_cdm))
        
        return m_star, m_cdm

class AstroDataModule(LightningDataModule):
    def __init__(
            self, 
            train_transforms=None, 
            test_transforms=None, 
            batch_size=1,
            num_workers=1,
            dataset='illustris'
        ):
        super().__init__()
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset

    def setup(self, stage=None):
        mass_mstar = None
        mass_cdm = None

        if stage == "fit" or stage is None:
            if self.dataset == "SIMBA":
                mass_mstar = np.load(
                    'Maps_Mstar_SIMBA_LH_z=0.00.npy'
                )
                mass_cdm = np.load(
                    'Maps_Mcdm_SIMBA_LH_z=0.00.npy'
                )
            elif self.dataset == "Astrid":
                mass_mstar = np.load(
                    'Maps_Mstar_Astrid_LH_z=0.00.npy'
                )
                mass_cdm = np.load(
                    'Maps_Mcdm_Astrid_LH_z=0.00.npy'
                )
            elif self.dataset=="IllustrisTNG":
                mass_mstar = np.load(
                    'Maps_Mstar_IllustrisTNG_LH_z=0.00.npy'
                )
                mass_cdm = np.load(
                    'Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy'
                )

            mass_mstar = Tensor(mass_mstar).unsqueeze(1)
            mass_cdm = Tensor(mass_cdm).unsqueeze(1)

            data = AstroDataset(mass_mstar, mass_cdm, transform=self.train_transforms)
            train_set_size = int(len(data) * 0.8)
            valid_set_size = len(data) - train_set_size
            generator = torch.Generator().manual_seed(342)
            self.train_data, self.valid_data = random_split(
                data, [train_set_size, valid_set_size], generator=generator
            )

        if stage == "test" or stage is None:
            mass_mstar = np.load(
                'Maps_Mstar_IllustrisTNG_1P_z=0.00.npy'
            )
            mass_cdm = np.load(
                'Maps_Mcdm_IllustrisTNG_1P_z=0.00.npy'
            )

            mass_mstar = Tensor(mass_mstar).unsqueeze(1)
            mass_cdm = Tensor(mass_cdm).unsqueeze(1)

            self.test_data = AstroDataset(
                mass_mstar, mass_cdm, transform=self.test_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=1, num_workers=self.num_workers
        )


def astro_normalizations(dataset):
    log_transform = transforms.Lambda(
        lambda x: (torch.log10(x[0] + 1), torch.log10(x[1]))
    )
    
    norm = Normalize(
        mean_input = norms[dataset][0],
        std_input= norms[dataset][1],
        mean_target= norms[dataset][2],
        std_target= norms[dataset][3]
    )

    return transforms.Compose([log_transform, norm])


def get_astro_data(dataset, num_workers=1, batch_size=10, stage=None, resize=None):
    train_transforms = [
            astro_normalizations(dataset),
            # Translate(),
            Flip(2),
            Permutate(2)
        ]

    test_transforms = [astro_normalizations(dataset)]

    if resize is not None:
        train_transforms += [Resize(resize)]
        test_transforms += [Resize(resize)]

    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    dm = AstroDataModule(
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        num_workers=num_workers,
        batch_size=batch_size,
        dataset=dataset
    )
    dm.setup(stage=stage)
    return dm