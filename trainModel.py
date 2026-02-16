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
    # Load data loaders, generators, discriminators, loss class
    train_loader, test_loader = getDataLoader(
        epoch_size=3000 if epoch_size is None else epoch_size
    )
    G_AB, G_BA = getGenerators()
    D_A, D_B = getDiscriminators()
    loss_fn = CycleGANLoss(lambda_cycle=10.0, lambda_identity=5.0)

    # Optimizers
    lr = 0.0002
    beta1 = 0.5

    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(beta1, 0.999)
    )

    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))

    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

    # Learning Rate Scheduler
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )

    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )

    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )

    # Full Training Loop
    torch.backends.cudnn.benchmark = True
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    history = {}

    model_dir = "data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models"
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(num_epochs):

        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()
        epochStep = {}

        for i, batch in enumerate(train_loader):

            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            # ==========================
            #  Train Generators
            # ==========================
            for p in D_A.parameters():
                p.requires_grad_(False)
            for p in D_B.parameters():
                p.requires_grad_(False)
            optimizer_G.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                loss_G, fake_A, fake_B = loss_fn.generator_loss(
                    real_A, real_B, G_AB, G_BA, D_A, D_B
                )

            scaler.scale(loss_G).backward()
            scaler.step(optimizer=optimizer_G)
            for p in D_A.parameters():
                p.requires_grad_(True)
            for p in D_B.parameters():
                p.requires_grad_(True)

            # ==========================
            #  Train Discriminator A
            # ==========================
            optimizer_D_A.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                loss_D_A = loss_fn.discriminator_loss(
                    D_A, real_A, fake_A, loss_fn.fake_A_buffer
                )
            scaler.scale(loss_D_A).backward()
            scaler.step(optimizer_D_A)

            # ==========================
            #  Train Discriminator B
            # ==========================
            optimizer_D_B.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                loss_D_B = loss_fn.discriminator_loss(
                    D_B, real_B, fake_B, loss_fn.fake_B_buffer
                )
            scaler.scale(loss_D_B).backward()
            scaler.step(optimizer_D_B)

            scaler.update()

            # ==========================
            #  Logging
            # ==========================
            epochStep[i + 1] = {
                "Batch": i + 1,
                "Loss_G": loss_G.item(),
                "Loss_D_A": loss_D_A.item(),
                "Loss_D_B": loss_D_B.item(),
            }
            if i % 500 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{i}/{len(train_loader)}] "
                    f"Loss_G: {loss_G.item():.4f} "
                    f"Loss_D_A: {loss_D_A.item():.4f} "
                    f"Loss_D_B: {loss_D_B.item():.4f}"
                )

        history[epoch + 1] = epochStep
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

        # Step LR schedulers
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    return history, G_AB, G_BA, D_A, D_B

if __name__ == "__main__":
    history, G_AB, G_BA, D_A, D_B = train(epoch_size=3000)