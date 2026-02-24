import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from data_loader import getDataLoader
from discriminator import getDiscriminators
from EarlyStopping import EarlyStopping
from generator import getGenerators
from history_utils import append_history_to_csv, load_history_from_csv
from losses import CycleGANLoss
from metrics import MetricsCalculator
from testing import run_testing
from validation import calculate_metrics, run_validation


def train(epoch_size=None, num_epochs=None, model_dir=None, val_dir=None):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_loader, test_loader = getDataLoader(
        epoch_size=3000 if epoch_size is None else epoch_size
    )
    G_AB, G_BA = getGenerators()
    D_A, D_B = getDiscriminators()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = CycleGANLoss(
        lambda_cycle=10.0,
        lambda_identity=5.0,
        lambda_cycle_perceptual=0.1,
        lambda_identity_perceptual=0.05,
        device=device,
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    metrics_calculator = MetricsCalculator(device=device)
    early_stopping = EarlyStopping(
        patience=10, min_delta=0.0001, divergence_threshold=5.0
    )

    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)

    lr = 0.0002
    beta1 = 0.5
    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(beta1, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )

    num_epochs = 200 if num_epochs is None else num_epochs
    history = {}

    model_dir = (
        "data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models"
        if model_dir is None
        else model_dir
    )
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}\\tensorboard_logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"{model_dir}\\tensorboard_logs")
    history_csv_path = os.path.join(model_dir, "training_history.csv")
    if os.path.exists(history_csv_path):
        os.remove(history_csv_path)

    for epoch in range(num_epochs):
        print("\n")

        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()
        epoch_step = {}

        writer.add_scalar("Epoch: ", epoch + 1, epoch + 1)

        for i, batch in enumerate(train_loader):
            i += 1

            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            for p in D_A.parameters():
                p.requires_grad_(False)
            for p in D_B.parameters():
                p.requires_grad_(False)
            optimizer_G.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                loss_G, fake_A, fake_B = loss_fn.generator_loss(
                    real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, num_epochs
                )

            scaler.scale(loss_G).backward()
            scaler.step(optimizer=optimizer_G)
            scaler.update()

            for p in D_A.parameters():
                p.requires_grad_(True)
            for p in D_B.parameters():
                p.requires_grad_(True)

            optimizer_D_A.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                loss_D_A = loss_fn.discriminator_loss(
                    D_A, real_A, fake_A, loss_fn.fake_A_buffer
                )
            scaler.scale(loss_D_A).backward()
            scaler.step(optimizer_D_A)
            scaler.update()

            optimizer_D_B.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                loss_D_B = loss_fn.discriminator_loss(
                    D_B, real_B, fake_B, loss_fn.fake_B_buffer
                )
            scaler.scale(loss_D_B).backward()
            scaler.step(optimizer_D_B)
            scaler.update()

            epoch_step[i] = {
                "Batch": i,
                "Loss_G": loss_G.item(),
                "Loss_D_A": loss_D_A.item(),
                "Loss_D_B": loss_D_B.item(),
            }

            if i == 1 or i == len(train_loader) or i % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{i}/{len(train_loader)}] "
                    f"Loss_G: {loss_G.item():.4f} "
                    f"Loss_D_A: {loss_D_A.item():.4f} "
                    f"Loss_D_B: {loss_D_B.item():.4f}"
                )

                global_step = epoch * len(train_loader) + i
                writer.add_scalar("Loss/Generator", loss_G.item(), global_step)
                writer.add_scalar("Loss/Discriminator_A", loss_D_A.item(), global_step)
                writer.add_scalar("Loss/Discriminator_B", loss_D_B.item(), global_step)

        history[epoch + 1] = epoch_step

        if (epoch + 1) % 5 == 0:
            append_history_to_csv(history, history_csv_path)
            history.clear()

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
            writer.add_scalar("Checkpoint saved", epoch + 1, epoch + 1)

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Learning Rate G: {lr_scheduler_G.get_last_lr()[0]:.6f} "
            f"Learning Rate D_A: {lr_scheduler_D_A.get_last_lr()[0]:.6f} "
            f"Learning Rate D_B: {lr_scheduler_D_B.get_last_lr()[0]:.6f}"
        )

        writer.add_scalar(
            "Learning Rate/Generator", lr_scheduler_G.get_last_lr()[0], epoch + 1
        )
        writer.add_scalar(
            "Learning Rate/Discriminator_A",
            lr_scheduler_D_A.get_last_lr()[0],
            epoch + 1,
        )
        writer.add_scalar(
            "Learning Rate/Discriminator_B",
            lr_scheduler_D_B.get_last_lr()[0],
            epoch + 1,
        )

        if val_dir is None:
            val_dir = f"{model_dir}\\validation_images"
        save_dir = os.path.join(val_dir, f"epoch_{epoch+1}")
        writer.add_scalar("Validation Started", epoch + 1, epoch + 1)
        run_validation(
            epoch=epoch + 1,
            G_AB=G_AB,
            G_BA=G_BA,
            test_loader=test_loader,
            device=device,
            save_dir=save_dir,
            num_samples=10,
            writer=writer,
        )

        if (epoch + 1) % 10 == 0:
            avg_metrics = calculate_metrics(
                calculator=metrics_calculator,
                G_AB=G_AB,
                G_BA=G_BA,
                test_loader=test_loader,
                device=device,
                writer=writer,
                epoch=epoch + 1,
            )
            print(f"Epoch {epoch + 1} Validation Metrics: {avg_metrics}")
            avg_ssim = (avg_metrics.get("ssim_A", 0) + avg_metrics.get("ssim_B", 0)) / 2
            avg_loss = (loss_G.item() + loss_D_A.item() + loss_D_B.item()) / 3

            if early_stopping(avg_ssim, avg_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("\n")
    calculate_metrics(
        calculator=metrics_calculator,
        G_AB=G_AB,
        G_BA=G_BA,
        test_loader=test_loader,
        device=device,
        writer=writer,
        epoch=num_epochs,
    )

    test_dir = os.path.join(model_dir, "test_images")
    writer.add_scalar("Testing Started", num_epochs, num_epochs)
    run_testing(
        G_AB=G_AB,
        G_BA=G_BA,
        test_loader=test_loader,
        device=device,
        save_dir=test_dir,
        writer=writer,
        epoch=num_epochs,
        num_samples=200,
    )

    writer.add_scalar("Training Completed", num_epochs, num_epochs)
    torch.save(
        {
            "epoch": num_epochs,
            "G_AB": G_AB.state_dict(),
            "G_BA": G_BA.state_dict(),
            "D_A": D_A.state_dict(),
            "D_B": D_B.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D_A": optimizer_D_A.state_dict(),
            "optimizer_D_B": optimizer_D_B.state_dict(),
        },
        f"{model_dir}\\final_checkpoint_epoch_{num_epochs}.pth",
    )

    append_history_to_csv(history, history_csv_path)
    history = load_history_from_csv(history_csv_path)

    writer.close()
    return history, G_AB, G_BA, D_A, D_B
