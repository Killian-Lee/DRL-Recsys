import torch
import torch.nn as nn
import torch.nn.functional as F

from virtualTB.utils import load_torch_file, package_file

class ActionModel(nn.Module):
    def __init__(self, n_input = 88 + 1 + 27, n_output = 11 + 10, learning_rate = 0.01):
        super(ActionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_output)
        )
        self.max_a = 11
        self.max_b = 10

    def predict(self, user, page, weight):
        x = self.model(torch.cat((user, page, weight), dim = -1))
        a = torch.multinomial(F.softmax(x[:, :self.max_a], dim = 1), 1)
        b = torch.multinomial(F.softmax(x[:, self.max_a:], dim = 1), 1)
        return torch.cat((a, b), dim = -1)

    def load(self, path = None):
        if path is None:
            with package_file("data", "action_model.pt") as model_path:
                state_dict = load_torch_file(model_path)
        else:
            state_dict = load_torch_file(path)
        self.model.load_state_dict(state_dict)
        return self
