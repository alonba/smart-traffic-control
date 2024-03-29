import torch
import pandas as pd
from brain.smart_net import SmartNet
import brain.hyper_params as hpam
from torch.utils.tensorboard import SummaryWriter

class SmartWriter(SummaryWriter):
    
    def __init__(self, log_dir: str):
        super().__init__('runs/' + log_dir)
        
    def weight_histograms_linear(self, step: int, weights: torch.tensor, layer_name: str) -> None:
        flattened_weights = weights.flatten()
        self.add_histogram(layer_name, flattened_weights, global_step=step, bins='tensorflow')
        
    def weight_histograms(self, smart_net: SmartNet, step: int) -> None:
        """
        Writes to TensorBoard the model's weights histogram
        """
        for agent in smart_net.agents.values():
            models = {'policy': agent.policy_net, 'target': agent.target_net}
            for model_name, model in models.items():
                
                # Iterate over all model layers
                for layer_number, param in enumerate(model.parameters()):
                    layer_name = f'{agent.name}/{model_name}/layer_{layer_number}'
                    self.weight_histograms_linear(step, param, layer_name)
                    
    def graphs(self, smart_net: SmartNet, state: dict):
        """
        Writes the nets structure to TensorBoard
        TODO fix - doesn't work with LSTM
        """
        for agent in smart_net.agents.values():
            agent_state = agent.filter_and_process_agent_state(state)
            model_input = [agent_state['own'], agent_state['neighbors']]
            if not hpam.SHARE_STATE:
                model_input[1] = torch.tensor([])
            if hpam.LSTM:
                agent.policy_net.requires_grad_(False)
                agent.target_net.requires_grad_(False)
            self.add_graph(agent.policy_net, input_to_model=model_input)
            self.add_graph(agent.target_net, input_to_model=model_input)
    
    def rewards_or_losses(self, smart_net: SmartNet, title: str, rewards_or_losses: pd.Series, episode: int):
        """
        Writes the total and individual rewards / losses to TensorBoard
        """
        self.add_scalar(f"Total{title}", rewards_or_losses.sum(), episode)
        for agent in smart_net.agents.values():
            self.add_scalar(f"{agent.name}/{title}", rewards_or_losses.loc[agent.name], episode)