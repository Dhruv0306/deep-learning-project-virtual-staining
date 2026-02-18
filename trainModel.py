# Imports
import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import torchvision
import os
from data_loader import getDataLoader, denormalize
from generator import getGenerators
from discriminator import getDiscriminators
from losses import CycleGANLoss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime


def train(epoch_size=None, num_epochs=None, model_dir=None, val_dir=None):
    """
    Train a CycleGAN model for image-to-image translation.

    This function implements the complete training loop for CycleGAN, including:
    - Loading data and models
    - Setting up optimizers and schedulers
    - Training generators and discriminators alternately
    - Saving checkpoints periodically

    Args:
        epoch_size (int, optional): Number of samples per epoch. Defaults to 3000 if None.
        num_epochs (int, optional): Total number of training epochs. Defaults to 200 if None.
        model_dir (str, optional): Directory path to save model checkpoints. Defaults to "models" if None.
        val_dir (str, optional): Validation directory path to save validation images

    Returns:
        tuple: A tuple containing:
            - history (dict): Training history with loss values for each epoch and batch
            - G_AB (torch.nn.Module): Generator for A->B translation
            - G_BA (torch.nn.Module): Generator for B->A translation
            - D_A (torch.nn.Module): Discriminator for domain A
            - D_B (torch.nn.Module): Discriminator for domain B
    """
    # Enable cuDNN benchmark mode for faster training on fixed input sizes
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Enable automatic mixed precision only for CUDA devices
    use_amp = device.type == "cuda"
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler("cuda", enabled=use_amp)

    # Move models to device
    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)

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
    # Total number of training epochs
    num_epochs = 200 if num_epochs is None else num_epochs
    # Dictionary to store training history
    history = {}

    # Create directory for saving model checkpoints
    model_dir = (
        "data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models"
        if model_dir is None
        else model_dir
    )
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
            scaler.update()

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
            scaler.update()

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
            # Print progress every 50 batches
            if i % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{i}/{len(train_loader)}] "
                    f"Loss_G: {loss_G.item():.4f} "
                    f"Loss_D_A: {loss_D_A.item():.4f} "
                    f"Loss_D_B: {loss_D_B.item():.4f}"
                )

        # Store epoch history
        history[epoch + 1] = epochStep
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
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

        # Run validation every epochs
        save_dir = f"{val_dir}\\epoch_{epoch+1}"
        run_validation(
            epoch=epoch + 1,
            G_AB=G_AB,
            G_BA=G_BA,
            test_loader=test_loader,
            device=device,
            save_dir=save_dir,
            num_samples=5,
        )

    return history, G_AB, G_BA, D_A, D_B


# Function to Validate model
def run_validation(epoch, G_AB, G_BA, test_loader, device, save_dir, num_samples=3):
    """
    Run validation for the CycleGAN model and save sample output images.

    This function evaluates the generators on the test dataset, generates translated
    images, and saves a few samples for visual inspection.

    Args:
        epoch (int): Current epoch number (used for naming saved images)
        G_AB (torch.nn.Module): Generator for A->B translation
        G_BA (torch.nn.Module): Generator for B->A translation
        test_loader (DataLoader): DataLoader for the test dataset
        device (torch.device): Device to run the validation on
        save_dir (str): Directory to save the generated sample images
        num_samples (int, optional): Number of sample images to save. Defaults to 3.
    """
    # Set generators to evaluation mode
    G_AB.eval()
    G_BA.eval()

    # Define validation loss
    total_cycle_loss = 0
    total_identity_loss = 0

    # Disable grad
    with torch.no_grad():
        # Create directory for saving validation images if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
            print(f"Validating Image {i}.")

            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            fake_B = G_AB(real_A)
            rec_A = G_BA(fake_B)
            fake_A = G_BA(real_B)
            rec_B = G_AB(fake_A)
            idt_A = G_BA(real_A)
            idt_B = G_AB(real_B)

            loss_idt_A = nn.L1Loss()(idt_A, real_A)
            loss_idt_B = nn.L1Loss()(idt_B, real_B)
            loss_cycle_A = nn.L1Loss()(rec_A, real_A)
            loss_cycle_B = nn.L1Loss()(rec_B, real_B)
            total_cycle_loss += loss_cycle_A.item() + loss_cycle_B.item()
            total_identity_loss += loss_idt_A.item() + loss_idt_B.item()

            # Save sample images for visual inspection
            # Save image as real_A | fack_B | rec_A | real_B as save_dir\\epoch_{epoch}_A.png
            # Save image as real_B | fack_A | rec_B | real_A as save_dir\\epoch_{epoch}_B.png
            save_validation_image(
                img_id=i + 1,
                real_A=real_A,
                fake_B=fake_B,
                rec_A=rec_A,
                real_B=real_B,
                fake_A=fake_A,
                rec_B=rec_B,
                epoch=epoch,
                save_dir=save_dir,
            )
    
    total_cycle_loss /= num_samples
    total_identity_loss /= num_samples
    print(f"Validation Epoch {epoch}: Average Cycle Loss: {total_cycle_loss:.4f}, Average Identity Loss: {total_identity_loss:.4f}")
    G_AB.train()
    G_BA.train()

# Function to save validation images
def save_validation_image(
    img_id, real_A, fake_B, rec_A, real_B, fake_A, rec_B, epoch, save_dir=None
):
    """
    Save a grid of images from a tensor.

    This function takes image tensors (real_A, fake_B, rec_A, etc.) and saves them as a single grid image.

    Args:
        real_A : real_A tensor
        fake_B : fake_B tensor
        rec_A : rec_A tensor
        real_B : real_B tensor
        fake_A : fake_A tensor
        rec_B : rec_B tensor
        epoch (int): Current epoch number (used for naming saved images)
        save_dir (str, optional): Directory to save the image
    """
    filename_A = (
        f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models\\validation_images\\epoch_{epoch}_A.png"
        if save_dir is None
        else f"{save_dir}\\image_{img_id}_A.png"
    )
    filename_B = (
        f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models\\validation_images\\epoch_{epoch}_B.png"
        if save_dir is None
        else f"{save_dir}\\image_{img_id}_B.png"
    )

    # Denormalize tensors
    real_A = denormalize(real_A[0]).permute(1, 2, 0).cpu().numpy()
    fake_B = denormalize(fake_B[0]).permute(1, 2, 0).cpu().numpy()
    rec_A = denormalize(rec_A[0]).permute(1, 2, 0).cpu().numpy()
    real_B = denormalize(real_B[0]).permute(1, 2, 0).cpu().numpy()
    fake_A = denormalize(fake_A[0]).permute(1, 2, 0).cpu().numpy()
    rec_B = denormalize(rec_B[0]).permute(1, 2, 0).cpu().numpy()

    # Save Figure real_A | fake_B | rec_A | real_B
    fig, axs = plt.subplots(1, 4, figsize=(28, 7))
    axs[0].imshow(real_A)
    axs[0].set_title("Real A")
    axs[1].imshow(fake_B)
    axs[1].set_title("Fake B")
    axs[2].imshow(rec_A)
    axs[2].set_title("Rec A")
    axs[3].imshow(real_B)
    axs[3].set_title("Real B")
    for ax in axs.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename_A)
    plt.close()

    # Save Figure real_B | fake_A | rec_B | real_A
    fig, axs = plt.subplots(1, 4, figsize=(28, 7))
    axs[0].imshow(real_B)
    axs[0].set_title("Real B")
    axs[1].imshow(fake_A)
    axs[1].set_title("Fake A")
    axs[2].imshow(rec_B)
    axs[2].set_title("Rec B")
    axs[3].imshow(real_A)
    axs[3].set_title("Real A")
    for ax in axs.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename_B)
    plt.close()
    print(f"Validation images saved.")


# Function to visualize training history and save it as a plot
def visualize_history(history, model_dir=None):
    """
    Visualize the training history of CycleGAN losses.

    Creates plots showing the progression of generator and discriminator losses
    over training epochs and batches.

    Args:
        history (dict): Training history dictionary with loss values for each epoch and batch
        model_dir (str, optional): Directory path where plots will be saved. Defaults to "models".
    """
    if not history:
        print("No training history to visualize.")
        return

    # Extract epoch numbers
    epochs = list(history.keys())

    # Initialize lists to store average losses per epoch
    avg_loss_G = []
    avg_loss_D_A = []
    avg_loss_D_B = []

    # Calculate average losses for each epoch
    for epoch in epochs:
        epoch_data = history[epoch]

        # Extract loss values for all batches in this epoch
        batch_loss_G = [batch_data["Loss_G"] for batch_data in epoch_data.values()]
        batch_loss_D_A = [batch_data["Loss_D_A"] for batch_data in epoch_data.values()]
        batch_loss_D_B = [batch_data["Loss_D_B"] for batch_data in epoch_data.values()]

        # Calculate averages
        avg_loss_G.append(np.mean(batch_loss_G))
        avg_loss_D_A.append(np.mean(batch_loss_D_A))
        avg_loss_D_B.append(np.mean(batch_loss_D_B))

    # Create subplots for different loss visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("CycleGAN Training History", fontsize=16)

    # Plot 1: All losses over epochs
    axes[0, 0].plot(
        epochs, avg_loss_G, label="Generator Loss", color="blue", linewidth=2
    )
    axes[0, 0].plot(
        epochs, avg_loss_D_A, label="Discriminator A Loss", color="red", linewidth=2
    )
    axes[0, 0].plot(
        epochs, avg_loss_D_B, label="Discriminator B Loss", color="green", linewidth=2
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Average Losses per Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Generator loss only
    axes[0, 1].plot(epochs, avg_loss_G, color="blue", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Generator Loss")
    axes[0, 1].set_title("Generator Loss Over Time")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Discriminator losses comparison
    axes[1, 0].plot(
        epochs, avg_loss_D_A, label="Discriminator A", color="red", linewidth=2
    )
    axes[1, 0].plot(
        epochs, avg_loss_D_B, label="Discriminator B", color="green", linewidth=2
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Discriminator Loss")
    axes[1, 0].set_title("Discriminator Losses Comparison")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Loss distribution for the last epoch (if available)
    if epochs:
        last_epoch = epochs[-1]
        last_epoch_data = history[last_epoch]

        batch_nums = list(last_epoch_data.keys())
        last_epoch_loss_G = [last_epoch_data[batch]["Loss_G"] for batch in batch_nums]
        last_epoch_loss_D_A = [
            last_epoch_data[batch]["Loss_D_A"] for batch in batch_nums
        ]
        last_epoch_loss_D_B = [
            last_epoch_data[batch]["Loss_D_B"] for batch in batch_nums
        ]

        axes[1, 1].plot(batch_nums, last_epoch_loss_G, label="Generator", alpha=0.7)
        axes[1, 1].plot(
            batch_nums, last_epoch_loss_D_A, label="Discriminator A", alpha=0.7
        )
        axes[1, 1].plot(
            batch_nums, last_epoch_loss_D_B, label="Discriminator B", alpha=0.7
        )
        axes[1, 1].set_xlabel("Batch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_title(f"Batch-wise Losses (Epoch {last_epoch})")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save before show to avoid blank images with interactive backends
    model_dir = (
        "data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models"
        if model_dir is None
        else model_dir
    )
    output_path = f"{model_dir}\\training_history.png"
    os.makedirs(model_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Training history plot saved to {output_path}")

    # Print summary statistics
    print("\n=== Training Summary ===")
    print(f"Total Epochs: {len(epochs)}")
    print(f"Final Generator Loss: {avg_loss_G[-1]:.4f}")
    print(f"Final Discriminator A Loss: {avg_loss_D_A[-1]:.4f}")
    print(f"Final Discriminator B Loss: {avg_loss_D_B[-1]:.4f}")
    print(f"Average Generator Loss: {np.mean(avg_loss_G):.4f}")
    print(f"Average Discriminator A Loss: {np.mean(avg_loss_D_A):.4f}")
    print(f"Average Discriminator B Loss: {np.mean(avg_loss_D_B):.4f}")

    plt.show()
    plt.close()


# Function to write back history to csv
def save_history_to_csv(history, filename):
    """
    Save the training history to a CSV file.

    This function converts the nested history dictionary into a flat DataFrame
    and saves it as a CSV file for further analysis.

    Args:
        history (dict): Training history dictionary with loss values for each epoch and batch
        filename (str): Path to the output CSV file
    """
    # Flatten the nested dictionary structure into a list of dictionaries
    flattened_data = []
    for epoch, batches in history.items():
        for batch, losses in batches.items():
            row = {"Epoch": epoch, "Batch": batch}
            row.update(losses)
            flattened_data.append(row)

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(flattened_data)
    df.to_csv(filename, index=False)
    print(f"\nHistory saved to {filename}")


# Main execution block - runs training when script is executed directly
if __name__ == "__main__":
    # Set up model directory with timestamp
    model_dir = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory: {model_dir}")

    # Create validation image directory
    val_dir = os.path.join(model_dir, "validation_images")
    os.makedirs(val_dir, exist_ok=True)
    print(f"Validation image directory: {val_dir}")

    # Take user input for epoch_size and num_epochs
    epoch_size = int(input("Enter Epoch Size: "))
    num_epochs = int(input("Enter Number of Epochs: "))

    # Start training with 3000 samples per epoch
    history, G_AB, G_BA, D_A, D_B = train(
        epoch_size=epoch_size, num_epochs=num_epochs, model_dir=model_dir, val_dir=val_dir
    )
    visualize_history(history, model_dir=model_dir)
    save_history_to_csv(
        history,
        f"{model_dir}\\training_history.csv",
    )
