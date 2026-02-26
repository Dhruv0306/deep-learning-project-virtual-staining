class EarlyStopping:
    def __init__(
        self,
        patience=10,
        min_delta=0.001,
        divergence_threshold=5.0,
        divergence_patience=2,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.divergence_threshold = divergence_threshold
        self.divergence_patience = divergence_patience
        self.best_ssim = -float("inf")
        self.counter = 0
        self.best_losses = {}
        self.divergence_counter = 0

    def __call__(self, ssim, losses):
        # Check for SSIM plateau
        if ssim > self.best_ssim + self.min_delta:
            self.best_ssim = ssim
            self.counter = 0
        else:
            self.counter += 1

        if not isinstance(losses, dict):
            losses = {"loss": float(losses)}

        diverged_losses = 0
        for name, value in losses.items():
            value = float(value)
            best = self.best_losses.get(name, float("inf"))

            # Initialize baseline per-loss before using divergence checks.
            if best == float("inf"):
                self.best_losses[name] = value
                continue

            if value > best * self.divergence_threshold:
                diverged_losses += 1

            if value < best:
                self.best_losses[name] = value

        if len(losses) > 0 and diverged_losses == len(losses):
            self.divergence_counter += 1
        else:
            self.divergence_counter = 0

        if self.divergence_counter >= self.divergence_patience:
            return True

        return self.counter >= self.patience
