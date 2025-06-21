
import torch
import torch.nn as nn
import torch.nn.init as init 

class QNet3(nn.Module):
    def __init__(self, config, num_lanes_map):
        super().__init__()

        self.config = config
        self.num_roads = len(num_lanes_map)
        self.num_lanes_map = num_lanes_map

        self._makeActionSize()
        self._makeNumFeatures()
        self.input_size = self.num_features
        self.output_size = self.action_size

        self.common_net = nn.Sequential(
            nn.Linear(self.num_features, 32),
            nn.ReLU(),
        )

        self.state_value_net = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Identity(),
        )

        self.advantage_net = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_size),
            nn.Identity(),
        )

        init.kaiming_normal_(self.common_net[0].weight, nonlinearity='relu')
        init.kaiming_normal_(self.advantage_net[0].weight, nonlinearity='relu')
        init.kaiming_normal_(self.advantage_net[2].weight, nonlinearity='linear')
        init.kaiming_normal_(self.state_value_net[0].weight, nonlinearity='relu')
        init.kaiming_normal_(self.state_value_net[2].weight, nonlinearity='linear')

    def set(self, key, value):
        setattr(self, key, value)
    
    def get(self, key):
        return getattr(self, key)

    def forward(self, x):
        common_output = self.common_net(x)
        state_value = self.state_value_net(common_output)
        advantage = self.advantage_net(common_output)
        q_values = state_value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def _makeNumFeatures(self):
        num_features = 0
        for _, num_lanes in self.num_lanes_map.items():
            num_features += num_lanes        

        self.num_features = num_features
    
    def _makeActionSize(self):
        num_roads_phases_map = self.config.get('num_roads_phases_map')
        phases_map = num_roads_phases_map[self.num_roads]
        num_phases = phases_map.shape[0]

        self.action_size = num_phases