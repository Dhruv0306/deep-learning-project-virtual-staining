import random
import torch

class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, tensors):
        output = []
        for tensor in tensors.detach():
            tensor = tensor.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(tensor)
                output.append(tensor)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    output.append(self.data[idx].clone())
                    self.data[idx] = tensor
                else:
                    output.append(tensor)
        return torch.cat(output)
