import torch
from torch import nn
import numpy as np

# augmentations
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform


########### main class #####################
def _modified_bn_forward(self, input):
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)

class Memo(nn.Module):
    def __init__(self, model, prior) -> None:
        super().__init__()
        self.model = model
        self.adapt_batchnorm(prior)
        self.augmentations = self.get_transforms()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

        # self.normalization = ZScoreNormalization()
  
    def adapt_batchnorm(self, prior_strength):
        nn.BatchNorm3d.prior = float(prior_strength) / float(prior_strength + 1)
        nn.BatchNorm3d.forward = _modified_bn_forward

    def forward(self, x):

        self.adapt_single(x)
        outputs = self.model(x)

        return outputs

    @torch.enable_grad() 
    def adapt_single(self, image):
        self.model.eval()
        inputs = [self.get_augmentations(image) for _ in range(4)] 
        inputs = torch.squeeze(torch.stack(inputs), dim = 1).cuda()

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, logits = self.marginal_entropy_3d(outputs)
        loss.backward()
        self.optimizer.step()

    def prep_for_aug(self, x_aug):
        data = {"image": torch.squeeze(x_aug).type(torch.float64).cpu()}
        return data
    
    def retrieve_from_aug(self, transformed):
        return transformed['image'].type(torch.float32).cuda()
    

    def get_augmentations(self, patch):
        """ it augments the input patch to have multiple augmentations"""        
        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(patch)
        for i in range(3):
            x_aug = patch.clone()
            for _ in range(np.random.randint(1, 4)):
                transformed = self.augmentations(**self.prep_for_aug(x_aug))  # Convert to numpy if needed
                x_aug = self.retrieve_from_aug(transformed)  # Convert back to tensor if necessary
            mix += w[i] * x_aug
        mix = m * patch + (1 - m) * mix
        return mix

    # original adapted to 3 dimensional network output
    def marginal_entropy_3d(self, outputs):
        # outputs shape: (N, C, D, H, W)

        # Flatten the spatial dimensions (D, H, W) into a single dimension for easier processing
        N, C, *spatial_dims = outputs.shape
        total_spatial_size = torch.tensor(spatial_dims, dtype=torch.float32, device=outputs.device).prod()  # Total number of spatial locations

        # Reshape to (N, C, spatial_size)
        outputs = outputs.view(N, C, -1)

        # Step 1: Normalize logits
        logits = outputs - outputs.logsumexp(dim=1, keepdim=True)  # Normalize across class dimension (C)

        # Step 2: Compute average logits over the batch and spatial dimensions
        avg_logits = logits.logsumexp(dim=0) - torch.log(torch.tensor(logits.shape[0], dtype=torch.float32, device=outputs.device))  # Average over N
        avg_logits = avg_logits.logsumexp(dim=-1) - torch.log(total_spatial_size)  # Average over spatial dims

        # Step 3: Clamp to prevent numerical instability
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)

        # Step 4: Compute marginal entropy
        entropy = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

        return entropy, avg_logits

    # intensity augmentations only for now

    def get_transforms(self, patch_size_spatial = [56, 192, 192]):

        transforms = []

        #values from nnunetTrainer
        rotation_for_DA = (-0.5235987755982988, 0.5235987755982988)
        mirror_axes = (0, 1, 2)
        ignore_axes = None

        transforms.append(RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.1),
                    p_per_channel=1,
                    synchronize_channels=True
                ), apply_probability=0.1
            ))
    
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
       
        return ComposeTransforms(transforms)
