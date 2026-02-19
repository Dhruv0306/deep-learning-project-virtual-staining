# Imports
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from PIL import Image


class MetricsCalculator:
    """
    A comprehensive metrics calculator for evaluating image generation and translation models.
    Supports SSIM, PSNR, and FID calculations for assessing image quality and distribution alignment.
    """

    def __init__(self, device=None):
        """
        Initialize the metrics calculator with necessary models and transforms.

        Args:
            device (str, Optional): Device to run computations on ('cuda' or 'cpu')
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Initialize InceptionV3 for FID calculation
        # InceptionV3 is commonly used for FID as it provides good feature representations
        self.inception_model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
        )
        setattr(
            self.inception_model, "fc", torch.nn.Identity()
        )  # Remove final layer to get features
        self.inception_model.eval().to(self.device)

        # Preprocessing for InceptionV3
        # InceptionV3 expects images normalized with ImageNet statistics
        self.inception_transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),  # InceptionV3 input size
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
        )

    def calculate_ssim(self, img1, img2):
        """
        Calculate SSIM (Structural Similarity Index) between two image tensors.
        SSIM measures structural similarity and is good for perceptual quality assessment.

        Args:
            img1, img2: Image tensors to compare

        Returns:
            float: SSIM value between -1 and 1 (higher is better)
        """
        # Convert to numpy and ensure proper format
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()

        # Handle batch dimension
        if len(img1.shape) == 4:
            ssim_values = []
            for i in range(img1.shape[0]):
                # Convert from CHW to HWC format expected by skimage
                im1 = np.transpose(img1[i], (1, 2, 0))
                im2 = np.transpose(img2[i], (1, 2, 0))
                # Calculate SSIM with multichannel support and appropriate data range
                ssim_val = ssim(im1, im2, channel_axis=-1, data_range=2.0)
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        else:
            # Single image case
            im1 = np.transpose(img1, (1, 2, 0))
            im2 = np.transpose(img2, (1, 2, 0))
            return ssim(im1, im2, channel_axis=-1, data_range=2.0)

    def calculate_psnr(self, img1, img2):
        """
        Calculate PSNR (Peak Signal-to-Noise Ratio) between two image tensors.
        PSNR measures reconstruction quality based on pixel-wise differences.

        Args:
            img1, img2: Image tensors to compare

        Returns:
            float: PSNR value in dB (higher is better)
        """
        # Calculate mean squared error
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float("inf")  # Perfect reconstruction
        # PSNR formula: 20 * log10(MAX_PIXEL_VALUE / sqrt(MSE))
        # Assuming pixel values in range [-1, 1], so MAX_PIXEL_VALUE = 2
        return 20 * torch.log10(2.0 / torch.sqrt(mse)).item()

    def get_inception_features(self, images):
        """
        Extract feature representations from InceptionV3 model.
        These features are used for FID calculation.

        Args:
            images: Batch of images in [-1, 1] range

        Returns:
            numpy.ndarray: Feature vectors from InceptionV3
        """
        with torch.no_grad():
            # Denormalize from [-1,1] to [0,1] then normalize for Inception
            images = (images + 1) / 2.0
            # Apply InceptionV3 preprocessing
            images = self.inception_transform(images)
            # Extract features (before final classification layer)
            features = self.inception_model(images)
        return features.cpu().numpy()

    def calculate_fid(self, real_images, fake_images):
        """
        Calculate FID (Fréchet Inception Distance) between real and fake images.
        FID measures the distance between feature distributions of real and generated images.
        Lower FID indicates better quality and diversity of generated images.

        Args:
            real_images: Batch of real images
            fake_images: Batch of generated/fake images

        Returns:
            float: FID score (lower is better)
        """
        # Get features from InceptionV3
        real_features = self.get_inception_features(real_images)
        fake_features = self.get_inception_features(fake_images)

        # Calculate statistics (mean and covariance) for both distributions
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

        # Calculate FID using the formula:
        # FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        diff = mu1 - mu2
        # Calculate matrix square root with regularization for numerical stability
        try:
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
        except:
            # Add small regularization to diagonal if matrix is singular
            eps = 1e-6
            sigma1_reg = sigma1 + eps * np.eye(sigma1.shape[0])
            sigma2_reg = sigma2 + eps * np.eye(sigma2.shape[0])
            covmean, _ = linalg.sqrtm(sigma1_reg.dot(sigma2_reg), disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real

        # Final FID calculation
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def evaluate_batch(self, real_A, real_B, fake_A, fake_B):
        """
        Evaluate a batch of images with multiple metrics.
        Useful for cycle-consistent models like CycleGAN.

        Args:
            real_A, real_B: Real images from domains A and B
            fake_A, fake_B: Generated images for domains A and B

        Returns:
            dict: Dictionary containing all calculated metrics
        """
        metrics = {}

        # SSIM for cycle consistency - measures structural similarity
        metrics["ssim_A"] = self.calculate_ssim(real_A, fake_A)
        metrics["ssim_B"] = self.calculate_ssim(real_B, fake_B)

        # PSNR for reconstruction quality - measures pixel-level accuracy
        metrics["psnr_A"] = self.calculate_psnr(real_A, fake_A)
        metrics["psnr_B"] = self.calculate_psnr(real_B, fake_B)

        return metrics

    def evaluate_fid(self, real_images, fake_images):
        """
        Calculate FID score for distribution alignment assessment.
        Wrapper function for FID calculation.

        Args:
            real_images: Real image distribution
            fake_images: Generated image distribution

        Returns:
            float: FID score
        """
        return self.calculate_fid(real_images, fake_images)
