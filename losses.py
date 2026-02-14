# Imports
import torch
import torch.nn as nn

# Full Loss Structure
# L_G = L_GAN + λ_cycle * L_cycle + λ_identity * L_identity


# CycleGAN Loss Class
class CycleGANLoss:
    def __init__(self, lambda_cycle=10.0, lambda_identity=5.0):
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        # Loss functions
        self.criterion_GAN = nn.MSELoss()  # LSGAN
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

    def generator_loss(self, real_A, real_B, G_AB, G_BA, D_A, D_B):

        # ------------------
        # Identity Loss
        # ------------------
        idt_A = G_BA(real_A)
        loss_idt_A = self.criterion_identity(idt_A, real_A) * self.lambda_identity

        idt_B = G_AB(real_B)
        loss_idt_B = self.criterion_identity(idt_B, real_B) * self.lambda_identity

        # ------------------
        # GAN Loss
        # ------------------
        fake_B = G_AB(real_A)
        pred_fake_B = D_B(fake_B)
        loss_GAN_AB = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

        fake_A = G_BA(real_B)
        pred_fake_A = D_A(fake_A)
        loss_GAN_BA = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

        # ------------------
        # Cycle Loss
        # ------------------
        rec_A = G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(rec_A, real_A) * self.lambda_cycle

        rec_B = G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(rec_B, real_B) * self.lambda_cycle

        # ------------------
        # Total Generator Loss
        # ------------------
        loss_G = (
            loss_GAN_AB
            + loss_GAN_BA
            + loss_cycle_A
            + loss_cycle_B
            + loss_idt_A
            + loss_idt_B
        )

        return loss_G, fake_A, fake_B

    def discriminator_loss(self, D, real, fake):

        # Real loss
        pred_real = D(real)
        loss_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

        # Fake loss
        pred_fake = D(fake.detach())
        loss_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        # Total
        loss_D = (loss_real + loss_fake) * 0.5

        return loss_D


if __name__ == "__main__":
    # Example usage
    loss_fn = CycleGANLoss(lambda_cycle=10.0, lambda_identity=5.0)
    print("CycleGAN Loss initialized successfully.")
    from generator import getGenerators
    from discriminator import getDiscriminators

    G_AB, G_BA = getGenerators()
    D_A, D_B = getDiscriminators()

    # Keep example tensors on the same device as the models.
    device = next(G_AB.parameters()).device
    real_A = torch.randn(1, 3, 256, 256, device=device)  # Example input
    real_B = torch.randn(1, 3, 256, 256, device=device)  # Example input

    loss_G, fake_A, fake_B = loss_fn.generator_loss(
        real_A, real_B, G_AB, G_BA, D_A, D_B
    )
    print("Generator loss:", loss_G)
    
    loss_D_A = loss_fn.discriminator_loss(D_A, real_A, fake_A)
    loss_D_B = loss_fn.discriminator_loss(D_B, real_B, fake_B)
    print("Discriminator A loss:", loss_D_A)
    print("Discriminator B loss:", loss_D_B)
