import torch
import torch.nn as nn
import torch.nn.functional as F

from virtualTB.utils import init_weight, load_torch_file, package_file

class LeaveModel(nn.Module):
    def __init__(self, n_input = 88, n_output = 101, learning_rate = 0.01):
        super(LeaveModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_output)
        )
        self.model.apply(init_weight)
        
    def predict(self, user):
        x = self.model(user)
        page = torch.multinomial(F.softmax(x, dim = 1), 1)
        return page

    def load(self, path = None):
        if path is None:
            with package_file("data", "leave_model.pt") as model_path:
                state_dict = load_torch_file(model_path)
        else:
            state_dict = load_torch_file(path)
        self.model.load_state_dict(state_dict)
        return self
