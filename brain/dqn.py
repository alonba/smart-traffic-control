import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions * 2)   # The 2 is for dividing the stay and advance to 2 different actions
        
    def forward(self, x):
        """
        Called with either one element to determine next action, or a batch during optimization.
        Returns tensor with 1 element
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # x = F.sigmoid(self.layer3(x))    # Needed only if we have only 1 output
        x = self.layer3(x)
        return x