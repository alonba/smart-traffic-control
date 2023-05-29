# from brain.smart_net import SmartNet
from torch.utils.tensorboard import SummaryWriter

class SmartWriter(SummaryWriter):
    
    def __init__(self):
        super().__init__()
        
    def weight_histograms_linear(self, step: int, weights, layer_name: str) -> None:
        flattened_weights = weights.flatten()
        self.add_histogram(layer_name, flattened_weights, global_step=step, bins='tensorflow')
        
    def weight_histograms(self, smart_net, step: int) -> None:
        """
        Writes to TensorBoard the model's weights histogram
        """
        for agent in smart_net.agents:
            models = {'policy': agent.policy_net, 'target': agent.target_net}
            for model_name, model in models.items():
                
                # Iterate over all model layers
                for layer_number, param in enumerate(model.parameters()):
                    layer_name = f'{agent.name}_{model_name}_layer_{layer_number}'
                    self.weight_histograms_linear(step, param, layer_name)
                    
    def graphs(self, smart_net, state):
        for agent in smart_net.agents:
            agent_obs = agent.filter_agent_obs_from_net_state(agent.name, state)
            agent_obs_tensor = agent.dict_vals_to_tensor(agent_obs)
            self.add_graph(agent.policy_net, agent_obs_tensor)
            self.add_graph(agent.target_net, agent_obs_tensor)