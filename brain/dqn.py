import torch
import torch.nn as nn
import torch.nn.functional as F
import brain.hyper_params as hpam

LSTM_LAYER = '_lstm_layer'
HIDDEN_TO_EMB_LAYER = '_hidden_to_emb_layer'

class DQN(nn.Module):

    def __init__(self, n_own_observations: int, n_neighbor_observations: dict, n_actions: int, neighbors: dict, device: str):
        """
        n_neighbor_observation - The number of observations each neighbor shares with the agent.
        The 2 outputs represent Q(s,stay) and Q(s,advance) where s is the input (state) to the network
        """
        super(DQN, self).__init__()
        
        # Determine the number of inputs to the simple (no LSTM) part of the NN
        input_layer_size = n_own_observations
        if hpam.LSTM:
            input_layer_size += hpam.EMBEDDING_DIM * len(neighbors)
        elif hpam.SHARE_STATE:
            input_layer_size += n_neighbor_observations['sum']
            
        if hpam.LSTM and hpam.IS_OWN_AGENT_STATE_EMBEDDING:
            self.own_state_encoder_layer = nn.Linear(n_own_observations, hpam.EMBEDDING_DIM, device=device)
            input_layer_size += hpam.EMBEDDING_DIM - n_own_observations
            
        self.layer1 = nn.Linear(input_layer_size, hpam.NET_WIDTH, device=device)
        self.layer2 = nn.Linear(hpam.NET_WIDTH, hpam.NET_WIDTH, device=device)
        self.layer3 = nn.Linear(hpam.NET_WIDTH, n_actions * 2, device=device)   # The 2 is used for having 2 outputs per action - advance / stay.
        
        # LSTM
        if hpam.LSTM:
            for neighbor in neighbors.keys():
                lstm_name = neighbor + LSTM_LAYER
                fc_name = neighbor + HIDDEN_TO_EMB_LAYER
                lstm = nn.LSTM(n_neighbor_observations[neighbor], hpam.HIDDEN_DIM, batch_first=True, device=device)
                fc = nn.Linear(hpam.HIDDEN_DIM, hpam.EMBEDDING_DIM, device=device)
                setattr(self, lstm_name, lstm)
                setattr(self, fc_name, fc)
        
    def forward(self, own_state: torch.Tensor, neighbors_state: dict) -> torch.Tensor:
        """
        # TODO add types to function declaration
        Called with either one element to determine next action, or a batch during optimization.
        Returns a tensor with 1 element
        """
        merged_state = self.forward_lstm_and_prepare_merged_state(own_state, neighbors_state)
        x = F.relu(self.layer1(merged_state))
        x = F.relu(self.layer2(x))
        net_output = self.layer3(x)
        return net_output
    
    def forward_lstm_and_prepare_merged_state(self, own_state: torch.Tensor, neighbors_state: dict) -> torch.Tensor:
        """
        Passes the inputs through LSTM if needed, and outputs the merged state, which is the input to the original fully-connected NN.
        lstm input - (batch, seq, feature)
        lstm output - (batch, seq, embed_dim)
        """
        merged_state = own_state
        if hpam.LSTM:
            # Run over merged state, with the encoded own state
            if hpam.IS_OWN_AGENT_STATE_EMBEDDING:
                merged_state = self.own_state_encoder_layer(own_state)
            
            # Put input into LSTM
            lstms_output = {}
            for neighbor in neighbors_state.keys():
                lstm_name = neighbor + LSTM_LAYER
                fc_name = neighbor + HIDDEN_TO_EMB_LAYER
                lstm = getattr(self, lstm_name)
                fc = getattr(self, fc_name)
                
                lstm_input = DQN.prepare_lstm_input(neighbors_state[neighbor])
                
                lstm_out, (lstm_hidden_n, lstm_cell_n) = lstm(lstm_input)
                lstms_output[neighbor] = fc(lstm_hidden_n)



            # lstms_output = {}
            # for neighbor in neighbors_state.keys():
            #     lstm_input = DQN.prepare_lstm_input(neighbors_state[neighbor])
            #     lstm_out, (lstm_hidden_n, lstm_cell_n) = self.lstms[neighbor]['lstm_layer'](lstm_input)
            #     lstms_output[neighbor] = self.lstms[neighbor]['hidden_to_emb_layer'](lstm_hidden_n)
                
            # Concat LSTM output to own_state
            for lstm_output in lstms_output.values():
                merged_state = torch.cat([merged_state, lstm_output.squeeze()], dim=-1)
        elif hpam.SHARE_STATE:
            for neighbor_state in neighbors_state.values():
                merged_state = torch.cat([merged_state, neighbor_state], dim=-1)
            
        return merged_state
    
    @staticmethod
    def prepare_lstm_input(state: torch.Tensor) -> torch.Tensor:
        """
        LSTM input needs to be of shape (batch_size, sequence_length, features_num)
        state of shape (num_of_features) is received when inferring.
        a batch tensor of shape (batch, seq, feature) is received on training.
        """
        lstm_input = state.view(1,1,-1) if (len(state.size()) == 1) else state
        return lstm_input