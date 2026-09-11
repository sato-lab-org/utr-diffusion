from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
from src.utils.utils import default, extract, linear_beta_schedule#, cosine_beta_schedule

# basically work same as diffusion(discrete version),
# available for continuously and masked multi-labels data where some labels are missing
class Diffusion_Masked_Continuous_Multi_Labels(nn.Module):
    def __init__(
        self,
        model,
        timestep: int = 50,
        beta_last: float = 0.2,
        condition_weight: float = 0.0,
        uncondition_prop: float = 0.1,
        label_wise_mask: bool = False,
    ):
        super().__init__()
        self.model = model
        self.timestep = timestep
        self.beta_last = beta_last
        self.cond_weight = condition_weight
        self.uncond_prop = uncondition_prop
        self.label_wise_mask = label_wise_mask
        # Diffusion params
        betas = linear_beta_schedule(timestep, beta_end=self.beta_last)
        #betas = cosine_beta_schedule(timestep)
        alphas = 1.0 - betas
        alphas_prod = torch.cumprod(alphas, dim=0)
        alphas_prod_prev = F.pad(alphas_prod[:-1], (1, 0), value=1.0)

        # Store as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_prod", alphas_prod)
        self.register_buffer("alphas_prod_prev", alphas_prod_prev)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_prod", torch.sqrt(alphas_prod))
        self.register_buffer("sqrt_1m_alphas_prod", torch.sqrt(1.0 - alphas_prod))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_prod_prev) / (1.0 - alphas_prod))

    @property
    def device(self):
        return self.betas.device

    @torch.no_grad()
    def sample(self, classes, shape, cond_weight, output_all_steps=False):
        return self.p_sample_loop(
            classes=classes,
            image_size=shape,
            cond_weight=cond_weight,
            output_all_steps=output_all_steps
        )

    @torch.no_grad()
    def sample_cross(self, classes, shape, cond_weight):
        return self.p_sample_loop(
            classes=classes,
            image_size=shape,
            cond_weight=cond_weight,
            get_cross_map=True,
        )

    @torch.no_grad()
    def p_sample_loop(self, classes, image_size, cond_weight, get_cross_map=False, output_all_steps=False):
        b = image_size[0]
        image = torch.randn(image_size, device=self.device)
        cross_images_final = []

        if classes is not None:
            uncond_mask = torch.cat([torch.ones_like(classes), torch.zeros_like(classes)], dim=0).to(self.device)
            classes = classes.repeat(2, 1)

            sampling_fn = partial(
                self.p_sample_guided,
                classes=classes,
                cond_weight=cond_weight,
                uncond_mask=uncond_mask,
            )
        else:
            sampling_fn = partial(self.p_sample)

        images = [image] if output_all_steps else None

        for i in reversed(range(self.timestep)):
            image = sampling_fn(x=image, t=torch.full((b,), i, device=self.device, dtype=torch.long), t_index=i)
            if output_all_steps:
                images.append(image)

        output = images if output_all_steps else image
        return (output, cross_images_final) if get_cross_map else output

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_1m_alphas_prod_t = extract(self.sqrt_1m_alphas_prod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, time=t) / sqrt_1m_alphas_prod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_guided(self, x, classes, t, t_index, uncond_mask, cond_weight):
        # adapted from: https://openreview.net/pdf?id=qw8AKxfYbI
        batch_size = x.shape[0]
        # double to do guidance with
        t_double = t.repeat(2).to(self.device)
        x_double = x.repeat(2, 1, 1, 1).to(self.device)
        betas_t = extract(self.betas, t_double, x_double.shape, self.device)
        sqrt_1m_alphas_prod_t = extract(self.sqrt_1m_alphas_prod, t_double, x_double.shape, self.device)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t_double, x_double.shape, self.device)

        label_is_presence = (~torch.isnan(classes)).to(self.device, dtype=torch.float32)
        classes = classes * uncond_mask
        label_is_presence = label_is_presence * uncond_mask

        # classifier free sampling interpolates between guided and non guided using `cond_weight`
        preds = self.model(x_double, time=t_double, classes=classes, label_is_presence=label_is_presence)
        eps1 = (1 + cond_weight) * preds[:batch_size]
        eps2 = cond_weight * preds[batch_size:]
        x_t = eps1 - eps2

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t[:batch_size] * (
            x - betas_t[:batch_size] * x_t / sqrt_1m_alphas_prod_t[:batch_size]
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape, self.device)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def reverse_process_guided(self, x_T, classes):
        sample_bz = x_T.shape[0]
        x_T_to_0,  x_t = [], x_T

        classes = classes.repeat(2) # double the batch
        context_mask = torch.ones_like(classes).to(self.device)
        context_mask[sample_bz:] = 0.0 # make last half of mask to unconditional
        sampling_fn = partial(
            self.p_sample_guided,
            classes=classes,
            cond_weight=self.cond_weight,
            context_mask=context_mask,
        )

        for i in reversed(range(0, self.timestep)):
            x_t, _ = sampling_fn(x=x_t, t=torch.full((sample_bz,), i, device=self.device, dtype=torch.long), t_index=i)
            x_T_to_0.append(x_t)

        return x_T_to_0

    def q_sample(self, x_start, t, noise=None):
        device = self.device
        noise = default(noise, torch.randn_like(x_start)).to(device)

        sqrt_alphas_prod_t = extract(self.sqrt_alphas_prod, t, x_start.shape, device)
        sqrt_1m_alphas_prod_t = extract(self.sqrt_1m_alphas_prod, t, x_start.shape, device)

        return sqrt_alphas_prod_t * x_start + sqrt_1m_alphas_prod_t * noise

    def q_sample_single_step(self, x_t_m1, t, noise=None):
        device = self.device
        noise = default(noise, torch.randn_like(x_t_m1,)).to(device)
        sqrt_beta = torch.sqrt(extract(self.betas, t, x_t_m1.shape, device))
        sqrt_alpha = torch.sqrt(1 - sqrt_beta)
        x_t = x_t_m1 * sqrt_alpha + noise * sqrt_beta
        return x_t

    def p_losses(self, x_start, t, classes, noise=None, loss_type="huber"):
        noise = default(noise, torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # turn nan to zero to partial conditional label
        label_is_presence = (~torch.isnan(classes)).to(self.device, dtype=torch.float32)
        classes = torch.nan_to_num(classes, nan=0.0) # [B, N]
        # mask for unconditional label for CFG
        if self.label_wise_mask:
            uncond_mask = torch.bernoulli(torch.zeros(classes.shape) + (1 - self.uncond_prop)).to(self.device)
        else:
            uncond_mask = torch.bernoulli(torch.zeros(classes.shape[0]) + (1 - self.uncond_prop)).to(self.device) # [B]
            uncond_mask = uncond_mask.unsqueeze(1).repeat(1, classes.shape[1]) # [B, N]
        classes = classes * uncond_mask
        label_is_presence = label_is_presence * uncond_mask
        # info of partial conditional label and unconditional label
        predicted_noise = self.model(x_noisy, time=t, classes=classes, label_is_presence=label_is_presence)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, classes):
        x = x.to(torch.float32)
        classes = classes.to(torch.float32)
        b = x.shape[0]
        t = torch.randint(0, self.timestep, (b,), device=self.device).long()

        return self.p_losses(x, t, classes)
