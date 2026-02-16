# Imports
import torch
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import os
from data_loader import getDataLoader
from generator import getGenerators
from discriminator import getDiscriminators
from losses import CycleGANLoss


def train(epoch_size=None):
    """
    Train a CycleGAN model for image-to-image translation.
    
    This function implements the complete training loop for CycleGAN, including:
    - Loading data and models
    - Setting up optimizers and schedulers
    - Training generators and discriminators alternately
    - Saving checkpoints periodically
    
    Args:
        epoch_size (int, optional): Number of samples per epoch. Defaults to 3000 if None.
        
    Returns:
        tuple: A tuple containing:
            - history (dict): Training history with loss values for each epoch and batch
            - G_AB (torch.nn.Module): Generator for A->B translation
            - G_BA (torch.nn.Module): Generator for B->A translation  
            - D_A (torch.nn.Module): Discriminator for domain A
            - D_B (torch.nn.Module): Discriminator for domain B
    """
    # Load data loaders, generators, discriminators, loss class
    # Get training and test data loaders with specified epoch size
    train_loader, test_loader = getDataLoader(
        epoch_size=3000 if epoch_size is None else epoch_size
    )
    # Initialize the two generators for bidirectional translation
    G_AB, G_BA = getGenerators()
    # Initialize discriminators for both domains
    D_A, D_B = getDiscriminators()
    # Initialize CycleGAN loss function with cycle and identity loss weights
    loss_fn = CycleGANLoss(lambda_cycle=10.0, lambda_identity=5.0)

    # Optimizers
    # Learning rate for all optimizers
    lr = 0.0002
    # Beta1 parameter for Adam optimizer (momentum term)
    beta1 = 0.5

    # Combined optimizer for both generators
    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(beta1, 0.999)
    )

    # Separate optimizer for discriminator A
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))

    # Separate optimizer for discriminator B
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

    # Learning Rate Scheduler
    # Linear decay scheduler for generator optimizer (starts decay after epoch 100)
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )

    # Linear decay scheduler for discriminator A optimizer
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )

    # Linear decay scheduler for discriminator B optimizer
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )

    # Full Training Loop
    # Enable cuDNN benchmark mode for faster training on fixed input sizes
    torch.backends.cudnn.benchmark = True
    # Total number of training epochs
    num_epochs = 200
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Enable automatic mixed precision only for CUDA devices
    use_amp = device.type == "cuda"
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler("cuda", enabled=use_amp)
    # Dictionary to store training history
    history = {}

    # Create directory for saving model checkpoints
    model_dir = "data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models"
    os.makedirs(model_dir, exist_ok=True)

    # Main training loop over epochs
    for epoch in range(num_epochs):

        # Set all models to training mode
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()
        # Dictionary to store loss values for current epoch
        epochStep = {}

        # Training loop over batches
        for i, batch in enumerate(train_loader):

            # Move real images from both domains to device
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            # ==========================
            #  Train Generators
            # ==========================
            # Freeze discriminator parameters during generator training
            for p in D_A.parameters():
                p.requires_grad_(False)
            for p in D_B.parameters():
                p.requires_grad_(False)
            # Clear generator gradients
            optimizer_G.zero_grad(set_to_none=True)

            # Forward pass through generators with mixed precision
            with autocast("cuda", enabled=use_amp):
                loss_G, fake_A, fake_B = loss_fn.generator_loss(
                    real_A, real_B, G_AB, G_BA, D_A, D_B
                )

            # Backward pass and optimizer step with gradient scaling
            scaler.scale(loss_G).backward()
            scaler.step(optimizer=optimizer_G)
            # Unfreeze discriminator parameters
            for p in D_A.parameters():
                p.requires_grad_(True)
            for p in D_B.parameters():
                p.requires_grad_(True)

            # ==========================
            #  Train Discriminator A
            # ==========================
            # Clear discriminator A gradients
            optimizer_D_A.zero_grad(set_to_none=True)

            # Forward pass through discriminator A with mixed precision
            with autocast("cuda", enabled=use_amp):
                loss_D_A = loss_fn.discriminator_loss(
                    D_A, real_A, fake_A, loss_fn.fake_A_buffer
                )
            # Backward pass and optimizer step
            scaler.scale(loss_D_A).backward()
            scaler.step(optimizer_D_A)

            # ==========================
            #  Train Discriminator B
            # ==========================
            # Clear discriminator B gradients
            optimizer_D_B.zero_grad(set_to_none=True)

            # Forward pass through discriminator B with mixed precision
            with autocast("cuda", enabled=use_amp):
                loss_D_B = loss_fn.discriminator_loss(
                    D_B, real_B, fake_B, loss_fn.fake_B_buffer
                )
            # Backward pass and optimizer step
            scaler.scale(loss_D_B).backward()
            scaler.step(optimizer_D_B)

            # Update gradient scaler for next iteration
            scaler.update()

            # ==========================
            #  Logging
            # ==========================
            # Store loss values for current batch
            epochStep[i + 1] = {
                "Batch": i + 1,
                "Loss_G": loss_G.item(),
                "Loss_D_A": loss_D_A.item(),
                "Loss_D_B": loss_D_B.item(),
            }
            # Print progress every 500 batches
            if i % 500 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{i}/{len(train_loader)}] "
                    f"Loss_G: {loss_G.item():.4f} "
                    f"Loss_D_A: {loss_D_A.item():.4f} "
                    f"Loss_D_B: {loss_D_B.item():.4f}"
                )

        # Store epoch history
        history[epoch + 1] = epochStep
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "G_AB": G_AB.state_dict(),
                    "G_BA": G_BA.state_dict(),
                    "D_A": D_A.state_dict(),
                    "D_B": D_B.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D_A": optimizer_D_A.state_dict(),
                    "optimizer_D_B": optimizer_D_B.state_dict(),
                },
                f"{model_dir}\\checkpoint_epoch_{epoch+1}.pth",
            )

        # Step LR schedulers at the end of each epoch
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    return history, G_AB, G_BA, D_A, D_B

# Main execution block - runs training when script is executed directly
if __name__ == "__main__":
    # Start training with 3000 samples per epoch
    history, G_AB, G_BA, D_A, D_B = train(epoch_size=3000)
