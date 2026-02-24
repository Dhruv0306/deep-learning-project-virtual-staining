import os

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image



def calculate_metrics(calculator, G_AB, G_BA, test_loader, device, writer, epoch):
    G_AB.eval()
    G_BA.eval()

    val_metrics = {"ssim_A": [], "ssim_B": [], "psnr_A": [], "psnr_B": []}
    real_B_list, fake_B_list = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 50:
                break

            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            batch_metrics = calculator.evaluate_batch(real_A, real_B, fake_A, fake_B)
            for key, value in batch_metrics.items():
                val_metrics[key].append(value)

            real_B_list.append(real_B)
            fake_B_list.append(fake_B)

    avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}

    if len(real_B_list) > 10:
        real_B_tensor = torch.cat(real_B_list[:10])
        fake_B_tensor = torch.cat(fake_B_list[:10])
        fid_score = calculator.evaluate_fid(real_B_tensor, fake_B_tensor)
        avg_metrics["fid"] = fid_score

    for metric_name, value in avg_metrics.items():
        writer.add_scalar(f"Validation/{metric_name}", value, epoch)

    print(
        f"Validation Metrics - SSIM_A: {avg_metrics['ssim_A']:.4f}, "
        f"SSIM_B: {avg_metrics['ssim_B']:.4f}, "
        f"PSNR_A: {avg_metrics['psnr_A']:.2f}, "
        f"PSNR_B: {avg_metrics['psnr_B']:.2f}"
    )

    if "fid" in avg_metrics:
        print(f"FID Score: {avg_metrics['fid']:.2f}")

    G_AB.train()
    G_BA.train()
    return avg_metrics


def run_validation(
    epoch, G_AB, G_BA, test_loader, device, save_dir, num_samples=3, writer=None
):
    G_AB.eval()
    G_BA.eval()

    total_cycle_loss = 0
    total_identity_loss = 0
    num_samples = min(num_samples, len(test_loader))
    num_samples = max(1, num_samples)

    os.makedirs(save_dir, exist_ok=True)

    idt_A_loss = nn.L1Loss()
    idt_B_loss = nn.L1Loss()
    cycle_A_loss = nn.L1Loss()
    cycle_B_loss = nn.L1Loss()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
            print(f"Validating Image {i}.")
            if writer is not None:
                writer.add_scalar("Validation Image", i + 1, epoch)

            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            fake_B = G_AB(real_A)
            rec_A = G_BA(fake_B)
            fake_A = G_BA(real_B)
            rec_B = G_AB(fake_A)
            idt_A = G_BA(real_A)
            idt_B = G_AB(real_B)

            loss_idt_A = idt_A_loss(idt_A, real_A)
            loss_idt_B = idt_B_loss(idt_B, real_B)
            loss_cycle_A = cycle_A_loss(rec_A, real_A)
            loss_cycle_B = cycle_B_loss(rec_B, real_B)
            total_cycle_loss += loss_cycle_A.item() + loss_cycle_B.item()
            total_identity_loss += loss_idt_A.item() + loss_idt_B.item()

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
    print(
        f"Validation Epoch {epoch}: Average Cycle Loss: {total_cycle_loss:.4f}, Average Identity Loss: {total_identity_loss:.4f}"
    )
    if writer is not None:
        writer.add_scalar("Validation/Average Cycle Loss", total_cycle_loss, epoch)
        writer.add_scalar(
            "Validation/Average Identity Loss", total_identity_loss, epoch
        )
    G_AB.train()
    G_BA.train()


def save_validation_image(
    img_id, real_A, fake_B, rec_A, real_B, fake_A, rec_B, epoch, save_dir=None
):
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

    row_A = torch.cat([real_A[:1], fake_B[:1], rec_A[:1], real_B[:1]], dim=0).detach().cpu()
    row_B = torch.cat([real_B[:1], fake_A[:1], rec_B[:1], real_A[:1]], dim=0).detach().cpu()

    grid_A = make_grid(row_A, nrow=4, normalize=True, value_range=(-1, 1))
    grid_B = make_grid(row_B, nrow=4, normalize=True, value_range=(-1, 1))

    save_image(grid_A, filename_A)
    save_image(grid_B, filename_B)
    print("Validation images saved.")
