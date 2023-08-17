import pandas
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd, Tensor
from typing import Optional, Tuple
from torch.special import expm1
from tqdm import trange
from torch.distributions.normal import Normal
from lightning.pytorch import LightningModule
from utils import FixedLinearSchedule, LearnedLinearSchedule, kl_std_normal
from validation import power


class VDM(nn.Module):
    def __init__(
        self,
        score_model: nn.Module,
        noise_schedule: str = "fixed_linear",
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        antithetic_time_sampling: bool = True,
        image_shape: Tuple[int] = (
            3,
            32,
            32,
        ),
        data_noise: float = 5.0e-4,
    ):
        """Variational diffusion model, continuous time implementation of arxiv:2107.00630.

        Args:
            score_model (nn.Module): model used to denoise
            noise_schedule (str, optional): whether fixed_linear or learned noise schedules.
            Defaults to "fixed_linear".
            gamma_min (float, optional): minimum gamma value. Defaults to -13.3.
            gamma_max (float, optional): maximum gamma value. Defaults to 5.0.
            antithetic_time_sampling (bool, optional): whether to do antithetic time sampling.
            Defaults to True.
            image_shape (Tuple[int], optional): image shape. Defaults to ( 3, 32, 32, ).
            data_noise (float, optional): noise in data, used for reconstruction loss.
            Defaults to 1.0e-3.

        Raises:
            ValueError: when noise_schedule not in (fixed_linear, learned_linear)
        """
        super().__init__()
        self.score_model = score_model
        self.image_shape = image_shape
        self.data_noise = data_noise
        if noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(gamma_min, gamma_max)
        elif noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(gamma_min, gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule {noise_schedule}")
        self.antithetic_time_sampling = antithetic_time_sampling

    def variance_preserving_map(
        self, x: Tensor, times: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """Add noise to data sample, in a variance preserving way Eq. 10 in arxiv:2107.00630

        Args:
            x (Tensor): data sample
            times (Tensor): time steps
            noise (Tensor, optional): noise to add. Defaults to None.

        Returns:
            Tensor: Noisy sample
        """
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            times = times.view(times.shape[0], 1, 1, 1)
            gamma_t = self.gamma(times)
        alpha = torch.sqrt(torch.sigmoid(-gamma_t))
        scale = torch.sqrt(torch.sigmoid(gamma_t))
        if noise is None:
            noise = torch.randn_like(x)
        return alpha * x + noise * scale, gamma_t

    def sample_times(
        self,
        batch_size: int,
        device: str,
    ) -> Tensor:
    
        """Sample diffusion times for batch, used for monte carlo estimates

        Args:
            batch_size (int): size of batch

        Returns:
            Tensor: times
        """
        if self.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=device)
        else:
            times = torch.rand(batch_size, device=device)
        return times

    def get_diffusion_loss(
        self,
        gamma_t: Tensor,
        times: Tensor,
        pred_noise: Tensor,
        noise: Tensor,
        bpd_factor: float,
    ) -> float:
        """get loss for diffusion process. Eq. 17 in arxiv:2107.00630

        Args:
            gamma_t (Tensor): gamma at time t
            times (Tensor): time steps
            pred_noise (Tensor): noise prediction
            noise (Tensor): noise added

        Returns:
            float: diffusion loss
        """
        gamma_grad = autograd.grad(  # gamma_grad shape: (B, )
            gamma_t,  # (B, )
            times,  # (B, )
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]
        pred_loss = (
            ((pred_noise - noise) ** 2).flatten(start_dim=1).sum(axis=-1)
        )  # (B, )
        return bpd_factor * 0.5 * pred_loss * gamma_grad

    def get_latent_loss(
        self,
        x: Tensor,
        bpd_factor: float,
    ) -> float:
        """Latent loss to ensure the prior is truly Gaussian

        Args:
            x (Tensor): data sample

        Returns:
            float: latent loss
        """
        gamma_1 = self.gamma(torch.tensor([1.0], device=x.device))
        sigma_1_sq = torch.sigmoid(gamma_1)
        mean_sq = (1 - sigma_1_sq) * x**2
        return bpd_factor * kl_std_normal(mean_sq, sigma_1_sq).flatten(start_dim=1).sum(
            axis=-1
        )

    def get_reconstruction_loss(
        self,
        x: Tensor,
        bpd_factor: float,
    ):
        """Measure reconstruction error

        Args:
            x (Tensor): data sample

        Returns:
            float: reconstruction loss
        """
        noise_0 = torch.randn_like(x)
        times = torch.tensor([0.0], device=x.device)
        z_0, gamma_0 = self.variance_preserving_map(
            x,
            times=times,
            noise=noise_0,
        )
        # Generate a sample for z_0 -> closest to the data
        alpha_0 = torch.sqrt(torch.sigmoid(-gamma_0))
        z_0_rescaled = z_0 / alpha_0
        return bpd_factor * Normal(loc=z_0_rescaled, scale=self.data_noise).log_prob(
            x
        ).flatten(start_dim=1).sum(axis=-1)

    def get_loss(
        self,
        x: Tensor,
        conditioning: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> float:
        """Get loss for diffusion model. Eq. 11 in arxiv:2107.00630

        Args:
            x (Tensor): data sample
            conditioning (Optional[Tensor], optional): conditioning. Defaults to None.
            noise (Optional[Tensor], optional): noise. Defaults to None.

        Returns:
            float: loss
        """
        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2))
        # Sample from q(x_t | x_0) with random t.
        times = self.sample_times(
            x.shape[0],
            device=x.device,
        ).requires_grad_(True)
        if noise is None:
            noise = torch.randn_like(x)
        x_t, gamma_t = self.variance_preserving_map(x=x, times=times, noise=noise)
        # Predict noise added
        pred_noise = self.score_model(
            x_t,
            conditioning=conditioning,
            g_t=gamma_t.squeeze(),
        )

        # *** Diffusion loss
        diffusion_loss = self.get_diffusion_loss(
            gamma_t=gamma_t,
            times=times,
            pred_noise=pred_noise,
            noise=noise,
            bpd_factor=bpd_factor,
        )

        # *** Latent loss: KL divergence from N(0, 1) to q(z_1 | x)
        latent_loss = self.get_latent_loss(
            x=x,
            bpd_factor=bpd_factor,
        )

        # *** Reconstruction loss:  - E_{q(z_0 | x)} [log p(x | z_0)].
        recons_loss = self.get_reconstruction_loss(
            x=x,
            bpd_factor=bpd_factor,
        )

        # *** Overall loss, Shape (B, ).
        loss = diffusion_loss + latent_loss + recons_loss

        metrics = {
            "elbo": loss.mean(),
            "diffusion_loss": diffusion_loss.mean(),
            "latent_loss": latent_loss.mean(),
            "reconstruction_loss": recons_loss.mean(),
        }
        return loss.mean(), metrics

    def alpha(self, gamma_t: Tensor) -> Tensor:
        """Eq. 4 arxiv:2107.00630

        Args:
            gamma_t (Tensor): gamma evaluated at t

        Returns:
            Tensor: alpha
        """
        return torch.sqrt(torch.sigmoid(-gamma_t))

    def sigma(self, gamma_t):
        """Eq. 3 arxiv:2107.00630

        Args:
            gamma_t (Tensor): gamma evaluated at t

        Returns:
            Tensor: sigma
        """
        return torch.sqrt(torch.sigmoid(gamma_t))

    def sample_zs_given_zt(
        self,
        zt: Tensor,
        conditioning: Tensor,
        t: Tensor,
        s: Tensor,
    ) -> Tensor:
        """Sample p(z_s|z_t, x) used for standard ancestral sampling. Eq. 34 in arxiv:2107.00630

        Args:
            z (Tensor): latent variable at time t
            conditioning (Tensor): conditioning for samples
            t (Tensor): time t
            s (Tensor): time s

        Returns:
            zs, samples for time s
        """
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = -expm1(gamma_s - gamma_t)
        alpha_t = self.alpha(gamma_t)
        alpha_s = self.alpha(gamma_s)
        sigma_t = self.sigma(gamma_t)
        sigma_s = self.sigma(gamma_s)
        pred_noise = self.score_model(
            zt,
            conditioning=conditioning,
            g_t=gamma_t,
        )
        mean = alpha_s / alpha_t * (zt - c * sigma_t * pred_noise)
        scale = sigma_s * torch.sqrt(c)
        return mean + scale * torch.randn_like(zt)

    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        batch_size: int,
        n_sampling_steps: int,
        device: str = "cpu",
        z: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate new samples given some conditioning vector

        Args:
            conditioning (Tensor): conditioning
            batch_size (int): number of samples in batch
            n_sampling_steps (int): number of sampling steps
            device (str, optional): device to run model. Defaults to "cpu".
            z (Optional[Tensor], optional): initial latent variable. Defaults to None.

        Returns:
            Tensor: generated sample
        """
        if z is None:
            z = torch.randn(
                (batch_size, *self.image_shape),
                device=device,
            )
        steps = torch.linspace(
            1.0,
            0.0,
            n_sampling_steps + 1,
            device=device,
        )
        
        for i in trange(n_sampling_steps, desc="sampling"):
            z = self.sample_zs_given_zt(
                zt=z,
                conditioning=conditioning,
                t=steps[i],
                s=steps[i + 1],
            )
        return z


class LightVDM(LightningModule):
    def __init__(
        self,
        score_model: nn.Module,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 1.0e-5,
        n_sampling_steps: int = 250,
        image_shape: Tuple[int] = (1, 256, 256),
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
    ):
        """Variational diffusion wrapper for lightning

        Args:
            score_model (nn.Module): model used to denoise
            learning_rate (float, optional): initial learning rate. Defaults to 1.0e-4.
            n_sampling_steps (int, optional): number of steps used to sample validations.
            Defaults to 250.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["score_model"])
        self.model = VDM(
            score_model=score_model,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            image_shape=image_shape,
        )

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        """get loss for samples

        Args:
            x (Tensor): data samples
            conditioning (Tensor): conditioning

        Returns:
            Tensor: loss
        """
        return self.model.get_loss(x=x, conditioning=conditioning)

    def evaluate(self, batch: Tuple, stage: str = None) -> Tensor:
        """get loss function

        Args:
            batch (Tuple): batch of examples
            stage (str, optional): training stage. Defaults to None.

        Returns:
            Tensor: loss function
        """
        conditioning, x = batch
        loss, metrics = self(x=x, conditioning=conditioning)
        if self.logger is not None:
            self.logger.log_metrics(metrics)
        return loss

    def training_step(
        self,
        batch: Tuple,
        batch_idx: int,
    ) -> Tensor:
        """Training

        Args:
            batch (Tuple): batch of examples
            batch_idx (int): batch idx

        Returns:
            Tensor: loss
        """
        return self.evaluate(batch, "train")

    def draw_samples(
        self,
        conditioning: Tensor,
        batch_size: int,
        n_sampling_steps: int,
    ) -> Tensor:
        """draw samples from model

        Args:
            conditioning (Tensor): conditioning
            batch_size (int): number of samples in batch
            n_sampling_steps (int): number of sampling steps used to generate validation
            samples

        Returns:
            Tensor: generated samples
        """

        return self.model.sample(
            conditioning=conditioning,
            batch_size=batch_size,
            n_sampling_steps=n_sampling_steps,
            device=conditioning.device,
        )

    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """validate model

        Args:
            batch (Tuple): batch of examples
            batch_idx (int): idx for batch

        Returns:
            Tensor: loss
        """
        # sample images during validation and upload to comet
        conditioning, x = batch
        if batch_idx == 0:
            sample = self.draw_samples(
                conditioning=conditioning,
                batch_size=len(x),
                n_sampling_steps=self.hparams.n_sampling_steps,
            )

            fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15,10))            
            ax.flat[0].imshow(conditioning[0].squeeze().cpu(), cmap='cividis')
            ax.flat[1].imshow(x[0].squeeze().cpu(), cmap='cividis', vmin=-3, vmax=3)
            ax.flat[2].imshow(sample[0].squeeze().cpu(), cmap='cividis', vmin=-3, vmax=3)
            ax.flat[0].set_title("Stars")
            ax.flat[1].set_title("True DM")
            ax.flat[2].set_title("Sampled DM")

            ax.flat[3].hist(x[0].cpu().numpy().flatten(), bins=np.linspace(-4, 4, 20), alpha=0.5, color='#4c4173', label="True DM")
            ax.flat[3].hist(sample[0].cpu().numpy().flatten(), bins=np.linspace(-4, 4, 20), alpha=0.5, color='#709bb5', label="Sampled DM")
            ax.flat[3].legend(fontsize=12)
            ax.flat[3].set_title("Density")

            k, P, N = power(x[0])
            ax.flat[4].loglog(k.cpu(), P.cpu(), label="True DM", color='#4c4173')
            k, P, N = power(sample[0])
            ax.flat[4].loglog(k.cpu(), P.cpu(), label="Sampled DM", color='#709bb5')
            ax.flat[4].legend(fontsize=12)
            ax.flat[4].set_xlabel('k',fontsize=15)
            ax.flat[4].set_ylabel('P(k)',fontsize=15)
            ax.flat[4].set_title("Power")

            ax.flat[5].imshow(sample[0].squeeze().cpu()-x[0].squeeze().cpu(), cmap='cividis')
            ax.flat[5].set_title("Residual field")

            if self.logger is not None:
                self.logger.experiment.log_figure(figure=plt,
                figure_name="Validation sample", step=self.current_epoch)
            plt.close()

        loss = self.evaluate(batch, "val")
        self.log("val_loss", loss)
            
        return loss

    def test_step(self, batch, batch_idx):
        """test model

        Args:
            batch (Tuple): batch of examples
            batch_idx (int): idx for batch

        Returns:
            Tensor: loss
        """
        conditioning, x = batch

        # compute power spectra averaged over the whole batch
        '''
        k_pred, P_true, P_pred = self.compute_power(conditioning, x, sample)
        if batch_idx == 65:
            torch.save(k_pred, 'debiasing/powers/dm/k_64.pt')

        torch.save(P_true, f'debiasing/powers/dm/true_128_batch_{batch_idx}.pt')
        torch.save(P_pred, f'debiasing/powers/dm/pred_128_batch_{batch_idx}.pt')
        '''

        # low and high parameter values
        indices = [5, 0, 10, 11, 21, 22, 32, 33, 43, 44, 54, 55, 65]
        titles = ['Typical', r'Low $\Omega_m$', r'High $\Omega_m$', r'Low $\sigma_8$', r'High $\sigma_8$', 
          r'Low $A_{SN1}$', r'High $A_{SN1}$', r'Low $A_{AGN1}$', r'High $A_{AGN1}$',
          r'Low $A_{SN2}$', r'High $A_{SN2}$', r'Low $A_{AGN2}$', r'High $A_{AGN2}$']
        
        if batch_idx in indices:
            sample = self.draw_samples(
            conditioning=conditioning,
            batch_size=len(x),
            n_sampling_steps=self.hparams.n_sampling_steps,
            )

            title = titles[indices.index(batch_idx)]

            fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))
            ax[0].imshow(conditioning[0].squeeze().cpu(), cmap='cividis', vmin=-.5, vmax=2)
            ax.flat[1].imshow(x[0].squeeze().cpu(), cmap='cividis', vmin=-3, vmax=3)
            ax.flat[2].imshow(sample[0].squeeze().cpu(), cmap='cividis', vmin=-3, vmax=3)
            ax.flat[0].set_title("Stars")
            ax.flat[1].set_title("True DM")
            ax.flat[2].set_title("Sampled DM")
            ax.flat[3].hist(x[0].cpu().numpy().flatten(), bins=np.linspace(-4, 4, 20), alpha=0.5, color='#e98d6b', label="True DM")
            ax.flat[3].hist(sample[0].cpu().numpy().flatten(), bins=np.linspace(-4, 4, 20), alpha=0.5, color='#b13c6c', label="Sampled DM")
            ax.flat[3].legend(fontsize=12)
            ax.flat[3].set_title("Density")
            k, P, N = power(x[0])
            ax.flat[4].loglog(k.cpu(), P.cpu(), label="True DM", color='#e98d6b')
            k, P, N = power(sample[0])
            ax.flat[4].loglog(k.cpu(), P.cpu(), label="Sampled DM", color='#b13c6c')
            ax.flat[4].legend(fontsize=12)
            ax.flat[4].set_xlabel('k',fontsize=15)
            ax.flat[4].set_ylabel(r'$P*k^2$',fontsize=15)
            ax.flat[4].set_title("Power")

            k, P, N = power(x)
            ax.flat[4].loglog(k.cpu(), P.cpu(), label="True DM", color='#e98d6b')
            k, P, N = power(sample)
            ax.flat[4].loglog(k.cpu(), P.cpu(), label="Sampled DM", color='#b13c6c')
            ax.flat[4].legend(fontsize=12)
            ax.flat[4].set_xlabel('k',fontsize=15)
            ax.flat[4].set_ylabel(r'$P*k^2$',fontsize=15)
            ax.flat[4].set_title("Averaged Power")

            fig.suptitle(title)

            if self.logger is not None:
                self.logger.experiment.log_figure(figure=plt,
                figure_name="1P sample")
                loss = F.mse_loss(sample, x)
                self.logger.log_metrics({"test_loss": loss})
            plt.close()

            
    def generate_samples(self, batch, batch_idx, batch_size, n=150):
        # batch_idx: 0 to 66
        # batch_size: 15
        conditioning, x = batch
        # could change conditioning in the argument in the future
        star = conditioning[0]
        star_fields = star.expand(batch_size, 
                                  star.shape[0], star.shape[1], star.shape[2])

        maps = [] # 10 tensors of shape ([15, 1, img_shape, img_shape])
        # draw n samples with the same conditioning
        for _ in range(n//batch_size):
            sample = self.draw_samples(
                conditioning=star_fields,
                batch_size=batch_size,
                n_sampling_steps=self.hparams.n_sampling_steps,
                )

            P_true, P_pred = self.compute_power(star_fields, x, sample)

            maps.append(sample)
        
        torch.save(maps, f'debiasing/maps/batch_{batch_idx}.pt')

        # plot 10 random samples
        # if batch_idx % 11 == 0:
        #     fig1, ax = plt.subplots(ncols=5, nrows=2)
        #     ax = ax.flatten()
        #     ax[0].imshow(x[0].squeeze().cpu(), cmap='cividis', vmin=-3, vmax=3)
        #     ax[0].set_axis_off()
        #     for i in range(9):
        #         j = np.random.randint(0, batch_size)
        #         ax[i+1].imshow(maps[i][j].squeeze().cpu(), cmap='cividis', vmin=-3, vmax=3)
        #         ax[i+1].set_axis_off()
        #     ax[0].set_title("True DM")

        #     post_mean = torch.mean(torch.mean(torch.stack(maps), axis=0), axis=0)
        #     post_var = torch.var(torch.var(torch.stack(maps), axis=0), axis=0)

        #     fig2, ax = plt.subplots(ncols=3)
        #     ax[0].imshow(x[0].squeeze().cpu(), cmap='cividis', vmin=-3, vmax=3)
        #     ax[0].set_title("True DM")
        #     ax[1].imshow(post_mean.squeeze().cpu(), cmap='cividis', vmin=-3, vmax=3)
        #     ax[1].set_title("Post. Mean")
        #     ax[2].imshow(post_var.squeeze().cpu(), cmap='cividis', vmin=-0.1, vmax=0.1)
        #     ax[2].set_title("Post. Var")

        #     if self.logger is not None:
        #         self.logger.experiment.log_figure(figure=fig1, figure_name='Generated sample images')
        #         self.logger.experiment.log_figure(figure=fig2, figure_name='Posterior mean and variance images')
        # plt.close()
        return maps


    def compute_power(self, conditioning, x, sample):
        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.3, 0.8, 0.6])
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])
        k_true, P_true, N = power(x[0])
        ax1.loglog(k_true.cpu(), P_true.cpu()*(k_true.cpu()**2), label="Star vs True DM", color='#4c4173')
        ax1.set_ylabel(r'$P*k^2$')
        k_pred, P_pred, N = power(sample[0])        
        ax1.loglog(k_pred.cpu(), P_pred.cpu()*(k_pred.cpu()**2), label="Star vs Sampled DM", color='#d16580')
        ax1.legend()
        ratio = P_pred / P_true
        ax2.plot(k_true.cpu(), ratio.cpu(), color='#935083')
        ax2.plot(k_true.cpu(), P_true.cpu()/P_true.cpu(), color='#fb8b6f')
        ax2.set_xscale('log')
        ax2.set_xlabel("k")
        ax2.set_ylabel("Ratio")
        ax2.set_ylim(0.75, 1.21)
        if self.logger is not None:
            self.logger.experiment.log_figure(figure=plt, figure_name='Generated sample power')
        plt.close()
        return k_pred, P_true, P_pred


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
        )
        return {'optimizer': optimizer, 
                'lr_scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 2}