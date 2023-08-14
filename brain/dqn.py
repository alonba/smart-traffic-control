import torch.nn as nn
import torch.nn.functional as F
import brain.hyper_params as hpam

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_neighbors = 0):
        """
        The 2 outputs represent Q(s,stay) and Q(s,advance) where s is the input (state) to the network
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hpam.NET_WIDTH)
        self.layer2 = nn.Linear(hpam.NET_WIDTH, hpam.NET_WIDTH)
        self.layer3 = nn.Linear(hpam.NET_WIDTH, n_actions * 2)   # This 2 is used for having 2 outputs per action - advance / stay.
        
        # LSTM
        if hpam.LSTM:
            self.lstm = nn.LSTM(hpam.K_STEPS_BACK, hpam.HIDDEN_DIM)
            self.hidden_to_embedding = nn.Linear(hpam.HIDDEN_DIM, hpam.EMBEDDING_DIM)
        
    def forward(self, x):
        """
        Called with either one element to determine next action, or a batch during optimization.
        Returns tensor with 1 element
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x