class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, divergence_threshold=5.0):
        self.patience = patience
        self.min_delta = min_delta
        self.divergence_threshold = divergence_threshold
        self.best_ssim = -float('inf')
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, ssim, loss):
        # Check for SSIM plateau
        if ssim > self.best_ssim + self.min_delta:
            self.best_ssim = ssim
            self.counter = 0
        else:
            self.counter += 1
            
        # Check for loss divergence
        if loss > self.best_loss * self.divergence_threshold:
            return True
            
        if loss < self.best_loss:
            self.best_loss = loss
            
        return self.counter >= self.patience
