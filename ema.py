
import torch 
import copy 

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for param_ema, param_model in zip(self.module.parameters(), model.parameters()):
                param_ema.data = self.decay * param_ema.data + (1 - self.decay) * param_model.data

    def __call__(self, x, active_b1ff=None):
        return self.module(x, active_b1ff) if active_b1ff is not None else self.module(x)
