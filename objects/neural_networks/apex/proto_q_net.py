from libs.torch_module import ExtendedModule

import torch
import torch.nn as nn

class ProtoQNet(ExtendedModule):
    def __init__(self, agent, requires_grad=True):
        super().__init__()
        
        if agent.__class__.__name__ == 'MasterAgent':
            self.master_agent = agent
        elif agent.__class__.__name__ == 'LocalAgent':
            self.local_agent = agent
            self.master_agent = self.local_agent.master_agent
        else:
            raise NotImplementedError(f"Not supported agent type: {agent.__class__.__name__}")
        
        self.config = agent.config
        self.device = agent.device

        self._initProps()
        self._makeNetwork()
        
        for param in self.parameters():
            param.requires_grad = requires_grad
        return
    
    def _initProps(self):
        self.num_roads = self.master_agent.get('num_roads')
        self.num_phases = self.master_agent.get('num_phases')
        self.num_lanes_map = self.master_agent.get('num_lanes_map')
        return
    
    def _makeNetwork(self):
        self.network_map = nn.ModuleDict()

        self.network_map['vehicle'] = VehicleNet(
            config=self.config, 
            num_roads=self.num_roads,
        )

        self.network_map['vehicles'] = VehiclesNet(
            config=self.config,
            num_vehicle_outputs=self.network_map['vehicle'].get('num_outputs'),
        )

        self.network_map['lane'] = LaneNet(
            config=self.config,
            num_vehicles_outputs=self.network_map['vehicles'].get('num_outputs'),
        )

        self.network_map['lanes'] = nn.ModuleDict()
        self.network_map['road'] = nn.ModuleDict()
        for road_id in range(1, self.num_roads + 1):
            num_lanes = self.num_lanes_map[road_id]

            if num_lanes in self.network_map['lanes']:
                continue

            self.network_map['lanes'][str(num_lanes)] = LanesNet(
                config=self.config,
                num_lane_outputs=self.network_map['lane'].get('num_outputs'),
                num_lanes=num_lanes,
            )

            self.network_map['road'][str(num_lanes)] = RoadNet(  
                config=self.config,
                num_lanes_outputs=self.network_map['lanes'][str(num_lanes)].get('num_outputs'),
            )
        
        self.network_map['roads'] = RoadsNet(
            config=self.config,
            num_road_outputs_map={road_id: self.network_map['road'][str(self.num_lanes_map[road_id])].get('num_outputs') for road_id in range(1, self.num_roads + 1)},
        )

        self.network_map['phase'] = PhaseNet(
            config=self.config,
            num_roads=self.num_roads,
        )

        self.network_map['intersection'] = IntersectionNet(
            config=self.config,
            num_roads_outputs=self.network_map['roads'].get('num_outputs'),
            num_phase_outputs=self.network_map['phase'].get('num_outputs'),
        )

        self.network_map['dueling'] = DuelingNet(
            config=self.config,
            num_intersection_outputs=self.network_map['intersection'].get('num_outputs'),
            num_roads=self.num_roads,
        )
        return

    def forward(self, states):
        # get phase_outputs (batch, phase features)
        phase_outputs = self.network_map['phase'](states['phase'])

        roads_inputs = []
        for road_id in range(1, self.num_roads + 1):
            lanes_inputs = []
            for lane_id in range(1, self.num_lanes_map[road_id] + 1):
                # get vehicles_outputs (batch, vehicles features)
                vehicle_outputs = self.network_map['vehicle'](states['roads'][f"road_{road_id}"]['lanes'][f"lane_{lane_id}"]['vehicles'])
                vehicle_outputs = vehicle_outputs.view(vehicle_outputs.size(0), -1)
                vehicles_outputs = self.network_map['vehicles'](vehicle_outputs)

                # get lane_outputs (batch, lane features)
                lane_inputs = torch.cat([
                    states['roads'][f"road_{road_id}"]['lanes'][f"lane_{lane_id}"]['lane'],
                    vehicles_outputs
                ], dim=1)
                lane_outputs = self.network_map['lane'](lane_inputs)
                lanes_inputs.append(lane_outputs)

            # get lanes_outputs (batch, lanes features)
            lanes_inputs = torch.cat(lanes_inputs, dim=1)
            lanes_outputs = self.network_map['lanes'][str(self.num_lanes_map[road_id])](lanes_inputs)

            road_inputs = torch.cat([
                lanes_outputs,
                states['roads'][f"road_{road_id}"]['road']
            ], dim=1)
            road_outputs = self.network_map['road'][str(self.num_lanes_map[road_id])](road_inputs)
            roads_inputs.append(road_outputs)
        
        # get roads_outputs (batch, roads features)
        roads_inputs = torch.cat(roads_inputs, dim=1)
        roads_outputs = self.network_map['roads'](roads_inputs)

        # get intersection_outputs (batch, intersection features)
        intersection_inputs = torch.cat([roads_outputs, phase_outputs], dim=1)
        intersection_outputs = self.network_map['intersection'](intersection_inputs)

        # get value_output and advantage_outputs (batch, num_actions)
        value_output, advantage_outputs = self.network_map['dueling'](intersection_outputs)

        # get q_values (batch, num_actions) 
        q_values = value_output + (advantage_outputs - advantage_outputs.mean(dim=1, keepdim=True))

        return q_values

class VehicleNet(ExtendedModule):
    def __init__(self, config, num_roads):
        super().__init__()

        self.config = config

        self._initProps(num_roads)
        self._makeNetwork()
        return

    def _initProps(self, num_roads):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['num_hidden_layers']

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']

        # get num_features
        num_features = self.config.get('num_features_map')['vehicle'][num_roads]

        # set num_input, num_output
        self.num_inputs = num_features
        self.num_outputs = int(num_features * self.compression_rate)

        # set num_hidden_layer_features_map
        self.num_hidden_layer_features_map = {}
        for layer_id in range(1, self.num_hidden_layers + 1):
            if self.compression_type == 'linear':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs - (layer_id * (self.num_inputs - self.num_outputs) / (self.num_hidden_layers + 1)))
            elif self.compression_type == 'geometric':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs * (self.num_outputs / self.num_inputs) ** (layer_id / (self.num_hidden_layers + 1)))
            else:
                raise NotImplementedError(f"Not supported compression type: {self.compression_type}")

        return
    
    def _makeNetwork(self):
        module_list = nn.ModuleList()

        if self.num_hidden_layers == 0:
            module_list.append(nn.Linear(self.num_inputs, self.num_outputs))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
            
            self.net = nn.Sequential(*module_list)
            return

        for layer_id in range(1, self.num_hidden_layers + 1):
            module_list.append(nn.Linear(
                self.num_inputs if layer_id == 1 else self.num_hidden_layer_features_map[layer_id - 1],
                self.num_hidden_layer_features_map[layer_id]
            ))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
    
        module_list.append(nn.Linear(
            self.num_hidden_layer_features_map[self.num_hidden_layers],
            self.num_outputs
        ))

        self.net = nn.Sequential(*module_list)
        return
    
    def forward(self, x):
        # xは（batch_size × num_vehicles × num_lanes × num_roads, num_features）のテンソル
        return self.net(x)
        
class VehiclesNet(ExtendedModule):
    def __init__(self, config, num_vehicle_outputs):
        super().__init__()

        self.config = config

        self._initProps(num_vehicle_outputs)
        self._makeNetwork()
        return

    def _initProps(self, num_vehicle_outputs):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['num_hidden_layers'] 

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']

        self.num_vehicles = drl_info['state']['vehicle']['number']

        # get num_features
        num_features = self.num_vehicles * num_vehicle_outputs

        # set num_input, num_output
        self.num_inputs = num_features
        self.num_outputs = int(num_features * self.compression_rate)

        # set num_hidden_layer_features_map
        self.num_hidden_layer_features_map = {}
        for layer_id in range(1, self.num_hidden_layers + 1):
            if self.compression_type == 'linear':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs - (layer_id * (self.num_inputs - self.num_outputs) / (self.num_hidden_layers + 1)))
            elif self.compression_type == 'geometric':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs * (self.num_outputs / self.num_inputs) ** (layer_id / (self.num_hidden_layers + 1)))
            else:
                raise NotImplementedError(f"Not supported compression type: {self.compression_type}")

        return
    
    def _makeNetwork(self):
        module_list = nn.ModuleList()
        if self.num_hidden_layers == 0:
            module_list.append(nn.Linear(self.num_inputs, self.num_outputs))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
            
            self.net = nn.Sequential(*module_list)
            return
        
        for layer_id in range(1, self.num_hidden_layers + 1):
            module_list.append(nn.Linear(
                self.num_inputs if layer_id == 1 else self.num_hidden_layer_features_map[layer_id - 1],
                self.num_hidden_layer_features_map[layer_id]
            ))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
            
        module_list.append(nn.Linear(
            self.num_hidden_layer_features_map[self.num_hidden_layers],
            self.num_outputs
        ))

        if self.activation_function == 'leaky_relu':
            module_list.append(nn.LeakyReLU(self.alpha))
        elif self.activation_function == 'relu':
            module_list.append(nn.ReLU())
        else:
            raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
        
        self.net = nn.Sequential(*module_list)
        return
    
    def forward(self, x):
        # xは（batch_size × num_lanes × num_roads, num_features）のテンソル
        return self.net(x)

class LaneNet(ExtendedModule):
    def __init__(self, config, num_vehicles_outputs):
        super().__init__()

        self.config = config

        self._initProps(num_vehicles_outputs)
        self._makeNetwork()
        return
    
    def _initProps(self, num_vehicles_outputs):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['num_hidden_layers'] 

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']

        # get num_features
        num_features = self.config.get('num_features_map')['lane']
        num_features += num_vehicles_outputs

        # set num_input, num_output
        self.num_inputs = num_features
        self.num_outputs = int(num_features * self.compression_rate)

        # set num_hidden_layer_features_map
        self.num_hidden_layer_features_map = {}
        for layer_id in range(1, self.num_hidden_layers + 1):
            if self.compression_type == 'linear':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs - (layer_id * (self.num_inputs - self.num_outputs) / (self.num_hidden_layers + 1)))
            elif self.compression_type == 'geometric':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs * (self.num_outputs / self.num_inputs) ** (layer_id / (self.num_hidden_layers + 1)))
            else:
                raise NotImplementedError(f"Not supported compression type: {self.compression_type}")
        return
    
    def _makeNetwork(self):
        module_list = nn.ModuleList()
        if self.num_hidden_layers == 0:
            module_list.append(nn.Linear(self.num_inputs, self.num_outputs))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
            
            self.net = nn.Sequential(*module_list)
            return
        
        for layer_id in range(1, self.num_hidden_layers + 1):
            module_list.append(nn.Linear(
                self.num_inputs if layer_id == 1 else self.num_hidden_layer_features_map[layer_id - 1],
                self.num_hidden_layer_features_map[layer_id]
            ))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")

        module_list.append(nn.Linear(
            self.num_hidden_layer_features_map[self.num_hidden_layers],
            self.num_outputs
        ))

        if self.activation_function == 'leaky_relu':
            module_list.append(nn.LeakyReLU(self.alpha))
        elif self.activation_function == 'relu':
            module_list.append(nn.ReLU())
        else:
            raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
        
        self.net = nn.Sequential(*module_list)
        return

    def forward(self, x):
        return self.net(x)
    
class LanesNet(ExtendedModule):
    def __init__(self, config, num_lane_outputs, num_lanes):
        super().__init__()

        self.config = config

        self._initProps(num_lane_outputs, num_lanes)
        self._makeNetwork()
        return
    
    def _initProps(self, num_lane_outputs, num_lanes):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['num_hidden_layers'] 

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']

        # get num_features
        num_features = num_lanes * num_lane_outputs

        # set num_input, num_output
        self.num_inputs = num_features
        self.num_outputs = int(num_features * self.compression_rate)

        # set num_hidden_layer_features_map
        self.num_hidden_layer_features_map = {}
        for layer_id in range(1, self.num_hidden_layers + 1):
            if self.compression_type == 'linear':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs - (layer_id * (self.num_inputs - self.num_outputs) / (self.num_hidden_layers + 1)))
            elif self.compression_type == 'geometric':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs * (self.num_outputs / self.num_inputs) ** (layer_id / (self.num_hidden_layers + 1)))
            else:
                raise NotImplementedError(f"Not supported compression type: {self.compression_type}")
        return
    
    def _makeNetwork(self):
        module_list = nn.ModuleList()
        if self.num_hidden_layers == 0:
            module_list.append(nn.Linear(self.num_inputs, self.num_outputs))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
            
            self.net = nn.Sequential(*module_list)
            return
        
        for layer_id in range(1, self.num_hidden_layers + 1):
            module_list.append(nn.Linear(
                self.num_inputs if layer_id == 1 else self.num_hidden_layer_features_map[layer_id - 1],
                self.num_hidden_layer_features_map[layer_id]
            ))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
            
        module_list.append(nn.Linear(
            self.num_hidden_layer_features_map[self.num_hidden_layers],
            self.num_outputs
        ))

        if self.activation_function == 'leaky_relu':
            module_list.append(nn.LeakyReLU(self.alpha))
        elif self.activation_function == 'relu':
            module_list.append(nn.ReLU())
        else:
            raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
        
        self.net = nn.Sequential(*module_list)
        return
    
    def forward(self, x):
        return self.net(x)

class RoadNet(ExtendedModule):
    def __init__(self, config, num_lanes_outputs):
        super().__init__()

        self.config = config

        self._initProps(num_lanes_outputs)
        self._makeNetwork()
        return

    def _initProps(self, num_lanes_outputs):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['num_hidden_layers'] 

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']

        # get num_features
        num_features = self.config.get('num_features_map')['road']
        num_features += num_lanes_outputs

        # set num_input, num_output
        self.num_inputs = num_features
        self.num_outputs = int(num_features * self.compression_rate)

        # set num_hidden_layer_features_map
        self.num_hidden_layer_features_map = {}
        for layer_id in range(1, self.num_hidden_layers + 1):
            if self.compression_type == 'linear':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs - (layer_id * (self.num_inputs - self.num_outputs) / (self.num_hidden_layers + 1)))
            elif self.compression_type == 'geometric':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs * (self.num_outputs / self.num_inputs) ** (layer_id / (self.num_hidden_layers + 1)))
            else:
                raise NotImplementedError(f"Not supported compression type: {self.compression_type}")
        return
    
    def _makeNetwork(self):
        module_list = nn.ModuleList()
        if self.num_hidden_layers == 0:
            module_list.append(nn.Linear(self.num_inputs, self.num_outputs))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
            
            self.net = nn.Sequential(*module_list)
            return
        
        for layer_id in range(1, self.num_hidden_layers + 1):
            module_list.append(nn.Linear(
                self.num_inputs if layer_id == 1 else self.num_hidden_layer_features_map[layer_id - 1],
                self.num_hidden_layer_features_map[layer_id]
            ))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")

        module_list.append(nn.Linear(
            self.num_hidden_layer_features_map[self.num_hidden_layers],
            self.num_outputs
        ))

        if self.activation_function == 'leaky_relu':
            module_list.append(nn.LeakyReLU(self.alpha))
        elif self.activation_function == 'relu':
            module_list.append(nn.ReLU())
        else:
            raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
        
        self.net = nn.Sequential(*module_list)
        return
    
    def forward(self, x):
        return self.net(x)
    
class RoadsNet(ExtendedModule):
    def __init__(self, config, num_road_outputs_map):
        super().__init__()

        self.config = config

        self._initProps(num_road_outputs_map)
        self._makeNetwork()
        return
    
    def _initProps(self, num_road_outputs_map):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['num_hidden_layers'] 

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']

        # get num_features
        num_features = sum(num_road_outputs_map.values())

        # set num_input, num_output
        self.num_inputs = num_features
        self.num_outputs = int(num_features * self.compression_rate)

        # set num_hidden_layer_features_map
        self.num_hidden_layer_features_map = {}
        for layer_id in range(1, self.num_hidden_layers + 1):
            if self.compression_type == 'linear':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs - (layer_id * (self.num_inputs - self.num_outputs) / (self.num_hidden_layers + 1)))
            elif self.compression_type == 'geometric':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs * (self.num_outputs / self.num_inputs) ** (layer_id / (self.num_hidden_layers + 1)))
            else:
                raise NotImplementedError(f"Not supported compression type: {self.compression_type}")
        return

    def _makeNetwork(self):
        module_list = nn.ModuleList()
        if self.num_hidden_layers == 0:
            module_list.append(nn.Linear(self.num_inputs, self.num_outputs))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
            
            self.net = nn.Sequential(*module_list)
            return
        
        for layer_id in range(1, self.num_hidden_layers + 1):
            module_list.append(nn.Linear(
                self.num_inputs if layer_id == 1 else self.num_hidden_layer_features_map[layer_id - 1],
                self.num_hidden_layer_features_map[layer_id]
            ))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
            
        module_list.append(nn.Linear(
            self.num_hidden_layer_features_map[self.num_hidden_layers],
            self.num_outputs
        ))

        if self.activation_function == 'leaky_relu':
            module_list.append(nn.LeakyReLU(self.alpha))
        elif self.activation_function == 'relu':
            module_list.append(nn.ReLU())
        else:
            raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
        
        self.net = nn.Sequential(*module_list)
        return
    
    def forward(self, x):
        return self.net(x)

class PhaseNet(ExtendedModule):
    def __init__(self, config, num_roads):
        super().__init__()

        self.config = config

        self._initProps(num_roads)
        self._makeNetwork()
        return
    
    def _initProps(self, num_roads):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['num_hidden_layers'] 

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']

        # get num_features
        num_features = self.config.get('num_features_map')['phase'][num_roads]

        # set num_input, num_output
        self.num_inputs = num_features
        self.num_outputs = int(num_features * self.compression_rate)

        # set num_hidden_layer_features_map
        self.num_hidden_layer_features_map = {}
        for layer_id in range(1, self.num_hidden_layers + 1):
            if self.compression_type == 'linear':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs - (layer_id * (self.num_inputs - self.num_outputs) / (self.num_hidden_layers + 1)))
            elif self.compression_type == 'geometric':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs * (self.num_outputs / self.num_inputs) ** (layer_id / (self.num_hidden_layers + 1)))
            else:
                raise NotImplementedError(f"Not supported compression type: {self.compression_type}")
        return
    
    def _makeNetwork(self):
        module_list = nn.ModuleList()
        if self.num_hidden_layers == 0:
            module_list.append(nn.Linear(self.num_inputs, self.num_outputs))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
            
            self.net = nn.Sequential(*module_list)
            return
        
        for layer_id in range(1, self.num_hidden_layers + 1):
            module_list.append(nn.Linear(
                self.num_inputs if layer_id == 1 else self.num_hidden_layer_features_map[layer_id - 1],
                self.num_hidden_layer_features_map[layer_id]
            ))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
        
        module_list.append(nn.Linear(
            self.num_hidden_layer_features_map[self.num_hidden_layers],
            self.num_outputs
        ))

        if self.activation_function == 'leaky_relu':
            module_list.append(nn.LeakyReLU(self.alpha))
        elif self.activation_function == 'relu':
            module_list.append(nn.ReLU())
        else:
            raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
        
        self.net = nn.Sequential(*module_list)
        return
    
    def forward(self, x):
        return self.net(x)
    
class IntersectionNet(ExtendedModule):
    def __init__(self, config, num_roads_outputs, num_phase_outputs):
        super().__init__()

        self.config = config

        self._initProps(num_roads_outputs, num_phase_outputs)
        self._makeNetwork()
        return
    
    def _initProps(self, num_roads_outputs, num_phase_outputs):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['num_hidden_layers'] 

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']

        # get num_features
        num_features = num_roads_outputs + num_phase_outputs

        # set num_input, num_output
        self.num_inputs = num_features
        self.num_outputs = int(num_features * self.compression_rate)

        # set num_hidden_layer_features_map
        self.num_hidden_layer_features_map = {}
        for layer_id in range(1, self.num_hidden_layers + 1):
            if self.compression_type == 'linear':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs - (layer_id * (self.num_inputs - self.num_outputs) / (self.num_hidden_layers + 1)))
            elif self.compression_type == 'geometric':
                self.num_hidden_layer_features_map[layer_id] = int(self.num_inputs * (self.num_outputs / self.num_inputs) ** (layer_id / (self.num_hidden_layers + 1)))
            else:
                raise NotImplementedError(f"Not supported compression type: {self.compression_type}")
        return
    
    def _makeNetwork(self):
        module_list = nn.ModuleList()
        if self.num_hidden_layers == 0:
            module_list.append(nn.Linear(self.num_inputs, self.num_outputs))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")

            self.net = nn.Sequential(*module_list)
            return

        for layer_id in range(1, self.num_hidden_layers + 1):
            module_list.append(nn.Linear(
                self.num_inputs if layer_id == 1 else self.num_hidden_layer_features_map[layer_id - 1],
                self.num_hidden_layer_features_map[layer_id]
            ))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
        
        module_list.append(nn.Linear(
            self.num_hidden_layer_features_map[self.num_hidden_layers], 
            self.num_outputs
        ))
        if self.activation_function == 'leaky_relu':
            module_list.append(nn.LeakyReLU(self.alpha))
        elif self.activation_function == 'relu':
            module_list.append(nn.ReLU())
        else:
            raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
        
        self.net = nn.Sequential(*module_list)
        return
    
    def forward(self, x):
        return self.net(x)
    
class DuelingNet(ExtendedModule):
    def __init__(self, config, num_intersection_outputs, num_roads):
        super().__init__()

        self.config = config

        self._initProps(num_intersection_outputs, num_roads)
        self._makeNetwork()
        return
    
    def _initProps(self, num_intersection_outputs, num_roads):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['num_hidden_layers'] 

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']

        # set num_input, num_output
        self.num_inputs = num_intersection_outputs
        self.num_outputs_map = {
            'value': 1,
            'advantage': self.config.get('num_features_map')['phase'][num_roads]
        } 

        # set num_hidden_layer_features_map
        self.num_hidden_layer_features_map = {'value': {}, 'advantage': {}}
        for part in ['value', 'advantage']:
            for layer_id in range(1, self.num_hidden_layers + 1):
                if self.compression_type == 'linear':
                    self.num_hidden_layer_features_map[part][layer_id] = int(self.num_inputs - (layer_id * (self.num_inputs - self.num_outputs_map[part]) / (self.num_hidden_layers + 1)))
                elif self.compression_type == 'geometric':
                    self.num_hidden_layer_features_map[part][layer_id] = int(self.num_inputs * (self.num_outputs_map[part] / self.num_inputs) ** (layer_id / (self.num_hidden_layers + 1)))
                else:
                    raise NotImplementedError(f"Not supported compression type: {self.compression_type}")

        return
    
    def _makeNetwork(self):
        self.network_map = nn.ModuleDict()

        
        if self.num_hidden_layers == 0:
            for part in ['value', 'advantage']:
                module_list = nn.ModuleList()   

                module_list.append(nn.Linear(self.num_inputs, self.num_outputs_map[part]))
                if self.activation_function == 'leaky_relu':
                    module_list.append(nn.LeakyReLU(self.alpha))
                elif self.activation_function == 'relu':
                    module_list.append(nn.ReLU())
                else:
                    raise NotImplementedError(f"Not supported activation function: {self.activation_function}")
                
                self.network_map[part] = nn.Sequential(*module_list)

            return
        
        for part in ['value', 'advantage']:
            module_list = nn.ModuleList()
            for layer_id in range(1, self.num_hidden_layers + 1):
                module_list.append(nn.Linear(
                    self.num_inputs if layer_id == 1 else self.num_hidden_layer_features_map[part][layer_id - 1],
                    self.num_hidden_layer_features_map[part][layer_id]
                ))

            if self.activation_function == 'leaky_relu':
                module_list.append(nn.LeakyReLU(self.alpha))
            elif self.activation_function == 'relu':
                module_list.append(nn.ReLU())
            else:
                raise NotImplementedError(f"Not supported activation function: {self.activation_function}")

            module_list.append(nn.Linear(
                self.num_hidden_layer_features_map[part][self.num_hidden_layers],
                self.num_outputs_map[part],
            ))

            self.network_map[part] = nn.Sequential(*module_list)
        return
    
    def forward(self, x):
        return self.network_map['value'](x), self.network_map['advantage'](x)
