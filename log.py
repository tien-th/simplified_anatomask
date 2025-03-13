# Create a logger class similar to the second code's nnUNetLogger
import wandb 

class Logger:
# Recording training losses ('train_losses')
# Tracking the start and end times of epochs ('epoch_start_timestamps' and 'epoch_end_timestamps')
# Calculating epoch duration for performance monitoring
    def __init__(self, experiment_name=None):
        wandb.login(key = '9ab49432fdba1dc80b8e9b71d7faca7e8b324e3e')
        wandb.init(project='FMMed', name=experiment_name)
        self.experiment_name = experiment_name
        self.logs_dict = {}

    def log(self, key, value):
            # wandb.log({key: value})
        self.logs_dict[key] = value

    def log_to_wandb(self):
        wandb.log(self.logs_dict)
        self.logs_dict = {}
