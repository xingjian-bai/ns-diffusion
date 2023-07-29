import sys
sys.path.append('../')
from utils import *
from dataset_clevr_ryan import BoundingBox
########################################################################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from random import random
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
from einops import rearrange, reduce


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.obj_denoise.channels != model.obj_denoise.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.obj_denoise.channels
        self.self_condition = self.model.obj_denoise.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, conds, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):

        with torch.enable_grad():
            model_output = self.model(conds, x, t, x_self_cond)
    
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
        # elif self.objective == 'pred_x0':
        #     x_start = model_output
        #     x_start = maybe_clip(x_start)
        #     pred_noise = self.predict_noise_from_start(x, t, x_start)

        # else if self.objective == 'pred_v':
        #     v = model_output
        #     x_start = self.predict_start_from_v(x, t, v)
        #     x_start = maybe_clip(x_start)
        #     pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, conds, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(conds, x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, conds, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(conds = conds, x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, conds, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(conds, img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    # @torch.inference_mode()
    # def ddim_sample(self, shape, return_all_timesteps = False):
    #     batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

    #     times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = list(reversed(times.int().tolist()))
    #     time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    #     img = torch.randn(shape, device = device)
    #     imgs = [img]

    #     x_start = None

    #     for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
    #         time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
    #         self_cond = x_start if self.self_condition else None
    #         pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

    #         if time_next < 0:
    #             img = x_start
    #             imgs.append(img)
    #             continue

    #         alpha = self.alphas_cumprod[time]
    #         alpha_next = self.alphas_cumprod[time_next]

    #         sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         c = (1 - alpha_next - sigma ** 2).sqrt()

    #         noise = torch.randn_like(img)

    #         img = x_start * alpha_next.sqrt() + \
    #               c * pred_noise + \
    #               sigma * noise

    #         imgs.append(img)

    #     ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    #     ret = self.unnormalize(ret)
    #     return ret

    @torch.inference_mode()
    def sample(self, conds, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        # if self.is_ddim_sampling:
            # raise ValueError(f'unknown objective {self.objective}')
        # conds, shape, return_all_timesteps = False

        generation_shape = (batch_size, channels, image_size, image_size)
        return self.p_sample_loop(conds, generation_shape, return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    # def interpolate(self, x1, x2, t = None, lam = 0.5):
    #     b, *_, device = *x1.shape, x1.device
    #     t = default(t, self.num_timesteps - 1)

    #     assert x1.shape == x2.shape

    #     t_batched = torch.full((b,), t, device = device)
    #     xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

    #     img = (1 - lam) * xt1 + lam * xt2

    #     x_start = None

    #     for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
    #         self_cond = x_start if self.self_condition else None
    #         img, x_start = self.p_sample(img, i, self_cond)

    #     return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, conds, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            raise ValueError(f"I don't deal with self conditioning")
            with torch.inference_mode():
                x_self_cond = self.model_predictions(conds, x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(conds, x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        # elif self.objective == 'pred_x0':
        #     target = x_start
        # if self.objective == 'pred_v':
        #     v = self.predict_v(x_start, t, noise)
        #     target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        ## ADD MASK HERE

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, conds, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(conds, img, t, *args, **kwargs)


# class GaussianDiffusion1D(nn.Module):
#     def __init__(
#         self,
#         model,
#         seq_length = 32,
#         obj_num = 2,
#         timesteps = 100,
#         sampling_timesteps = None,
#         objective = 'pred_noise',
#         beta_schedule = 'cosine',
#         ddim_sampling_eta = 0.,
#         auto_normalize = True,
#     ):
#         super().__init__()
#         self.model = model


#         self.out_shape = (obj_num, 4)
#         self.self_condition = False

#         self.seq_length = seq_length
#         self.objective = objective
#         assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

#         if beta_schedule == 'linear':
#             betas = linear_beta_schedule(timesteps)
#         elif beta_schedule == 'cosine':
#             betas = cosine_beta_schedule(timesteps)
#         else:
#             raise ValueError(f'unknown beta schedule {beta_schedule}')

#         alphas = 1. - betas
#         alphas_cumprod = torch.cumprod(alphas, dim=0)
#         alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

#         timesteps, = betas.shape
#         self.num_timesteps = int(timesteps)

#         # sampling related parameters

#         self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

#         assert self.sampling_timesteps <= timesteps
#         self.is_ddim_sampling = self.sampling_timesteps < timesteps
#         self.ddim_sampling_eta = ddim_sampling_eta

#         # helper function to register buffer from float64 to float32

#         register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

#         register_buffer('betas', betas)
#         register_buffer('alphas_cumprod', alphas_cumprod)
#         register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

#         # calculations for diffusion q(x_t | x_{t-1}) and others

#         register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
#         register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
#         register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
#         register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
#         register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

#         # Step size for optimizing
#         register_buffer('opt_step_size', 0.3 * betas * torch.sqrt( 1 / (1 - alphas_cumprod)))

#         # calculations for posterior q(x_{t-1} | x_t, x_0)

#         posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

#         # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

#         register_buffer('posterior_variance', posterior_variance)

#         # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

#         register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
#         register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
#         register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

#         # calculate loss weight

#         snr = alphas_cumprod / (1 - alphas_cumprod)

#         if objective == 'pred_noise':
#             loss_weight = torch.ones_like(snr)
#         elif objective == 'pred_x0':
#             loss_weight = snr
#         elif objective == 'pred_v':
#             loss_weight = snr / (snr + 1)

#         register_buffer('loss_weight', loss_weight)
#         # whether to autonormalize


#     def predict_start_from_noise(self, x_t, t, noise):
#         """
#         Predict x_0 from x_t and noise
#         """
#         return (
#             extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
#             extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
#         )

#     def predict_noise_from_start(self, x_t, t, x0):
#         return (
#             (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
#             extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
#         )

#     def predict_v(self, x_start, t, noise):
#         return (
#             extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
#             extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
#         )

#     def predict_start_from_v(self, x_t, t, v):
#         return (
#             extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
#             extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
#         )

#     def q_posterior(self, x_start, x_t, t):
#         """
#         This is the posterior distribution q(x_{t-1} | x_t, x_0)
#         """
#         posterior_mean = (
#             extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
#             extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
#         )
#         posterior_variance = extract(self.posterior_variance, t, x_t.shape)
#         posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
#         return posterior_mean, posterior_variance, posterior_log_variance_clipped

#     def model_predictions(self, conds, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
#         """
#         Returns model predictions for the given conditional inputs and timesteps.
#         """
#         with torch.enable_grad():
#             model_output = self.model(conds, x, t)

#         maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

#         if self.objective == 'pred_noise':
#             pred_noise = model_output
#             x_start = self.predict_start_from_noise(x, t, pred_noise)
#             x_start = maybe_clip(x_start)

#             if clip_x_start and rederive_pred_noise:
#                 pred_noise = self.predict_noise_from_start(x, t, x_start)
#         # elif self.objective == 'pred_x0':
#         #     x_start = model_output
#         #     x_start = maybe_clip(x_start)
#         #     pred_noise = self.predict_noise_from_start(x, t, x_start)

#         # elif self.objective == 'pred_v':
#         #     v = model_output
#         #     x_start = self.predict_start_from_v(x, t, v)
#         #     x_start = maybe_clip(x_start)
#         #     pred_noise = self.predict_noise_from_start(x, t, x_start)
#         else:
#             raise ValueError(f'Unknown objective {self.objective}')

#         return ModelPrediction(pred_noise, x_start)

#     def p_mean_variance(self, conds, x, t, x_self_cond = None, clip_denoised = False):
#         preds = self.model_predictions(conds, x, t, x_self_cond)
#         x_start = preds.pred_x_start

#         if clip_denoised:
#             x_start.clamp_(-5., 5.)

#         model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
#         return model_mean, posterior_variance, posterior_log_variance, x_start

#     @torch.no_grad()
#     def p_sample(self, conds, x, t: int, x_self_cond = None, clip_denoised = True):
#         b, *_, device = *x.shape, x.device
#         batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
#         model_mean, _, model_log_variance, x_start = self.p_mean_variance(conds, x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
#         noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
#         pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

#         return pred_img, x_start

#     @torch.no_grad()
#     def p_sample_loop(self, batch_size, shape, inp, cond, mask):
#         device = self.betas.device

#         if not hasattr(self.model, 'randn'):
#             img = torch.randn((batch_size, *shape), device=device)
#         else:
#             raise NotImplementedError
#             img = self.model.randn(batch_size, shape, inp, device)
            

#         x_start = None

#         if mask is not None:
#             img = img * (1 - mask) + cond * mask

#         for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps, disable = True):
#             self_cond = x_start if self.self_condition else None

#             img, x_start = self.p_sample(inp, img, t, self_cond)

#             if mask is not None:
#                 img = img * (1 - mask) + cond * mask

#             # if t < 50:
#             # batched_times = torch.full((img.shape[0],), t, device = inp.device, dtype = torch.long)

#         return img

#     @torch.no_grad()
#     def ddim_sample(self, shape, clip_denoised = True):
#         raise NotImplementedError
#         batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

#         times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
#         times = list(reversed(times.int().tolist()))
#         time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

#         img = torch.randn(shape, device = device)

#         x_start = None

#         for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
#             time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
#             self_cond = x_start if self.self_condition else None
#             pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)

#             if time_next < 0:
#                 img = x_start
#                 continue

#             alpha = self.alphas_cumprod[time]
#             alpha_next = self.alphas_cumprod[time_next]

#             sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
#             c = (1 - alpha_next - sigma ** 2).sqrt()

#             noise = torch.randn_like(img)

#             img = x_start * alpha_next.sqrt() + \
#                   c * pred_noise + \
#                   sigma * noise

#         return img

#     @torch.no_grad()
#     def sample(self, x, label, mask, batch_size = 16):
#         # seq_length, channels = self.seq_length, self.channels
#         sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
#         return sample_fn(batch_size, self.out_shape, x, label, mask)

#     # @torch.no_grad()
#     # def interpolate(self, x1, x2, t = None, lam = 0.5):
#     #     b, *_, device = *x1.shape, x1.device
#     #     t = default(t, self.num_timesteps - 1)

#     #     assert x1.shape == x2.shape

#     #     t_batched = torch.full((b,), t, device = device)
#     #     xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

#     #     img = (1 - lam) * xt1 + lam * xt2

#     #     x_start = None

#     #     for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
#     #         self_cond = x_start if self.self_condition else None
#     #         img, x_start = self.p_sample(img, i, self_cond)

#     #     return img

#     def q_sample(self, x_start, t, noise=None):
#         noise = default(noise, lambda: torch.randn_like(x_start))

#         return (
#             extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
#             extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
#         )

#     def p_losses(self, inp, x_start, mask, t, noise = None):
#         b, *c = x_start.shape
#         noise = default(noise, lambda: torch.randn_like(x_start))

#         # noise sample
#         x = self.q_sample(x_start = x_start, t = t, noise = noise)

#         if mask is not None:
#             # Mask out inputs
#             x = x * (1 - mask) + mask * x_start

#         # predict and take gradient step

#         model_out = self.model(inp, x, t)

#         if self.objective == 'pred_noise':
#             target = noise
#         # elif self.objective == 'pred_x0':
#         #     target = x_start
#         # elif self.objective == 'pred_v':
#         #     v = self.predict_v(x_start, t, noise)
#         #     target = v
#         else:
#             raise ValueError(f'unknown objective {self.objective}')

#         if mask is not None:
#             # Mask out targets
#             model_out = model_out * (1 - mask) + mask * target

#         loss = F.mse_loss(model_out, target, reduction = 'none')
#         loss = reduce(loss, 'b ... -> b (...)', 'mean')

#         loss = loss * extract(self.loss_weight, t, loss.shape)
#         loss_mse = loss

#         loss = loss_mse # + 1e-2 * loss_energy
#         return loss.mean(), (loss_mse.mean(), -1)

#     def forward(self, inp, target, mask, *args, **kwargs):
#         b, *c = target.shape
#         device = target.device
#         if len(c) == 1:
#             self.out_shape = c
#         else:
#             self.out_shape = c

#         t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

#         return self.p_losses(inp, target, mask, t, *args, **kwargs)