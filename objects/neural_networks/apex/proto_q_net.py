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

        if agent.__class__.__name__ == 'MasterAgent':
            self._showInfo('parameters')
        
        if self.init_weights_flg:
            self.apply(self._initWeights)
            self._initOptimisticWeights()
        
        for param in self.parameters():
            param.requires_grad = requires_grad
        return
    
    def _initProps(self):
        # set information from master_agent
        self.num_roads = self.master_agent.get('num_roads')
        self.num_phases = self.master_agent.get('num_phases')
        self.num_lanes_map = self.master_agent.get('num_lanes_map')
        
        # set network initialization information
        drl_info = self.config.get('drl_info')
        self.init_weights_flg = drl_info['architecture']['common']['initialization']['type'] is not None
        if self.init_weights_flg:
            self.init_weights_type = drl_info['architecture']['common']['initialization']['type']
            self.activation_function = drl_info['architecture']['common']['activation_function']['type']
            if self.activation_function == 'leaky_relu':
                self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']

        return
    
    def _makeNetwork(self):
        self.network_map = nn.ModuleDict()

        self.network_map['vehicle_encoder'] = VehicleEncoderNet(
            config=self.config, 
            num_roads=self.num_roads,
        )

        self.network_map['vehicles'] = VehiclesNet(
            config=self.config,
            num_vehicle_outputs=self.network_map['vehicle_encoder'].get('num_outputs'),
        )

        self.network_map['lane_encoder'] = LaneEncoderNet(
            config=self.config,
        )

        self.network_map['lane'] = LaneNet(
            config=self.config,
            num_vehicles_outputs=self.network_map['vehicles'].get('num_outputs'),
            num_lane_encoder_outputs=self.network_map['lane_encoder'].get('num_outputs'),
        )

        self.network_map['lanes'] = nn.ModuleDict()
        for road_id in range(1, self.num_roads + 1):
            num_lanes = self.num_lanes_map[road_id]

            if num_lanes in self.network_map['lanes']:
                continue

            self.network_map['lanes'][str(num_lanes)] = LanesNet(
                config=self.config,
                num_lane_outputs=self.network_map['lane'].get('num_outputs'),
                num_lanes=num_lanes,
            )
        
        self.network_map['road_encoder'] = RoadEncoderNet(
            config=self.config,
            num_roads=self.num_roads,
        )
        
        self.network_map['road'] = nn.ModuleDict()
        for road_id in range(1, self.num_roads + 1):
            num_lanes = self.num_lanes_map[road_id]

            if num_lanes in self.network_map['road']:
                continue

            self.network_map['road'][str(num_lanes)] = RoadNet(  
                config=self.config,
                num_lanes=num_lanes,
                num_lanes_outputs=self.network_map['lanes'][str(num_lanes)].get('num_outputs'),
                num_road_encoder_outputs=self.network_map['road_encoder'].get('num_outputs'),
            )
        
        self.network_map['roads'] = RoadsNet(
            config=self.config,
            num_road_outputs_map={road_id: self.network_map['road'][str(self.num_lanes_map[road_id])].get('num_outputs') for road_id in range(1, self.num_roads + 1)},
        )

        self.network_map['intersection_encoder'] = IntersectionEncoderNet(
            config=self.config,
            num_roads=self.num_roads,
        )

        self.network_map['intersection'] = IntersectionNet(
            config=self.config,
            num_roads_outputs=self.network_map['roads'].get('num_outputs'),
            num_phase_outputs=self.network_map['intersection_encoder'].get('num_outputs'),
        )

        self.network_map['dueling'] = DuelingNet(
            config=self.config,
            num_intersection_outputs=self.network_map['intersection'].get('num_outputs'),
            num_roads=self.num_roads,
        )
        return
    
    def _initWeights(self, module):
        if not isinstance(module, nn.Linear):
            return
        
        if self.init_weights_type == 'xavier':
            nn.init.xavier_uniform_(module.weight)

        elif self.init_weights_type == 'he':
            if self.activation_function == 'relu':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif self.activation_function == 'leaky_relu':
                nn.init.kaiming_uniform_(module.weight, a=self.alpha, nonlinearity='leaky_relu')

        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)

        return
    
    def _initOptimisticWeights(self):
        for part in ['value', 'advantage']:
            final_layer = self.network_map['dueling'].network_map[part][-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.constant_(final_layer.bias, 10.0)
        return
    
    def _showInfo(self, type):
        print('==============================================')
        if type == 'gradient':
            print('status: check gradients')
            print(f"master agent id: {self.master_agent.get('id')}")
            
            for sub_network_name in self.network_map:
                if sub_network_name in ['lanes', 'road']:
                    for num_lanes in self.network_map[sub_network_name]:
                        self.network_map[sub_network_name][num_lanes].showInfo('gradient')
                else:
                    self.network_map[sub_network_name].showInfo('gradient')

        elif type == 'parameters':
            print('status: number of model parameters')
            print(f"master agent id: {self.master_agent.get('id')}")

            for sub_network_name in self.network_map:
                if sub_network_name in ['lanes', 'road']:
                    for num_lanes in self.network_map[sub_network_name]:
                        self.network_map[sub_network_name][num_lanes].showInfo('parameters')
                else:
                    self.network_map[sub_network_name].showInfo('parameters')

            total_params = sum(p.numel() for p in self.parameters())
            print(f"total parameters: {total_params}")

        else:
            raise NotImplementedError(f"Not supported type: {type}")

        return

    def forward(self, states):
        roads_inputs = []
        for road_id in range(1, self.num_roads + 1):
            lanes_inputs = []
            for lane_id in range(1, self.num_lanes_map[road_id] + 1):
                # get vehicles_outputs (batch, vehicles features)
                vehicle_outputs = self.network_map['vehicle_encoder'](states['roads'][f"road_{road_id}"]['lanes'][f"lane_{lane_id}"]['vehicles'])
                vehicle_outputs = vehicle_outputs.view(vehicle_outputs.size(0), -1)
                vehicles_outputs = self.network_map['vehicles'](vehicle_outputs)

                # get lane_encoder_outputs (batch, lane_encoder features)
                lane_encoder_outputs = self.network_map['lane_encoder'](states['roads'][f"road_{road_id}"]['lanes'][f"lane_{lane_id}"]['lane'])

                # get lane_outputs (batch, lane features)
                lane_inputs = torch.cat([
                    vehicles_outputs, 
                    lane_encoder_outputs
                ], dim=1)
                lane_outputs = self.network_map['lane'](lane_inputs)

                # append lane_outputs to lanes_inputs
                lanes_inputs.append(lane_outputs)

            # get lanes_outputs (batch, lanes features)
            lanes_inputs = torch.cat(lanes_inputs, dim=1)
            lanes_outputs = self.network_map['lanes'][str(self.num_lanes_map[road_id])](lanes_inputs)

            # get road_encoder_outputs (batch, road_encoder features)
            road_encoder_outputs = self.network_map['road_encoder'](states['roads'][f"road_{road_id}"]['road'])

            # get road_outputs (batch, road features)
            road_inputs = torch.cat([
                lanes_outputs,
                road_encoder_outputs
            ], dim=1)
            road_outputs = self.network_map['road'][str(self.num_lanes_map[road_id])](road_inputs)

            # append road_outputs to roads_inputs
            roads_inputs.append(road_outputs)
        
        # get roads_outputs (batch, roads features)
        roads_inputs = torch.cat(roads_inputs, dim=1)
        roads_outputs = self.network_map['roads'](roads_inputs)

        # get intersection_encoder_outputs (batch, phase features)
        intersection_encoder_outputs = self.network_map['intersection_encoder'](states['intersection'])

        # get intersection_outputs (batch, intersection features)
        intersection_inputs = torch.cat([roads_outputs, intersection_encoder_outputs], dim=1)
        intersection_outputs = self.network_map['intersection'](intersection_inputs)

        # get value_output and advantage_outputs (batch, num_actions)
        value_output, advantage_outputs = self.network_map['dueling'](intersection_outputs)

        # get q_values (batch, num_actions) 
        q_values = value_output + (advantage_outputs - advantage_outputs.mean(dim=1, keepdim=True))

        return q_values
    
    def showInfo(self, type):
        self._showInfo(type)
        return
    
class VehicleEncoderNet(ExtendedModule):
    def __init__(self, config, num_roads):
        super().__init__()

        self.config = config

        self._initProps(num_roads)
        self._makeNetwork()
        return

    def _initProps(self, num_roads):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['hidden_layers']['number']['vehicle_encoder']

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']['vehicle_encoder']

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
    
    def _showInfo(self, type):
        if type == 'gradient':
            print(f"vehicle network:")

            counter = 1
            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                print(f"linear layer {counter}: weight = {layer.weight.grad.norm().item():.3f}, bias = {layer.bias.grad.norm().item():.3f}")
                counter += 1
        
        elif type == 'parameters':
            num_params = sum(p.numel() for p in self.parameters())
            print(f"vehicle network parameters: {num_params}")
            
        else:
            raise NotImplementedError(f"Not supported type: {type}")

        return
    
    def forward(self, x):
        # xは（batch_size × num_vehicles × num_lanes × num_roads, num_features）のテンソル
        return self.net(x)
    
    def showInfo(self, type):
        self._showInfo(type)
        return
        
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

        self.num_hidden_layers = drl_info['architecture']['proto']['hidden_layers']['number']['vehicles']

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']['vehicles']

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
    
    def _showInfo(self, type):
        if type == 'gradient':
            print(f"vehicles network:")

            counter = 1
            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                print(f"linear layer {counter}: weight = {layer.weight.grad.norm().item():.3f}, bias = {layer.bias.grad.norm().item():.3f}")
                counter += 1

        elif type == 'parameters':
            num_params = sum(p.numel() for p in self.parameters())
            print(f"vehicles network parameters: {num_params}")
        
        else:
            raise NotImplementedError(f"Not supported type: {type}")

        return
    
    def forward(self, x):
        # xは（batch_size × num_lanes × num_roads, num_features）のテンソル
        return self.net(x)
    
    def showInfo(self, type):
        self._showInfo(type)
        return  
    
class LaneEncoderNet(ExtendedModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self._initProps()
        self._makeNetwork()
        return
    
    def _initProps(self):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['hidden_layers']['number']['lane_encoder']

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']

        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']['lane_encoder']

        # get num_features
        num_features = self.config.get('num_features_map')['lane']

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
    
    def _showInfo(self, type):
        if type == 'gradient':
            print(f"lane encoder network:")

            counter = 1
            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                print(f"linear layer {counter}: weight = {layer.weight.grad.norm().item():.3f}, bias = {layer.bias.grad.norm().item():.3f}")
                counter += 1

        elif type == 'parameters':
            num_params = sum(p.numel() for p in self.parameters())
            print(f"lane encoder network parameters: {num_params}")
        
        else:
            raise NotImplementedError(f"Not supported type: {type}")

        return  
    
    def forward(self, x):
        return self.net(x)
    
    def showInfo(self, type):
        self._showInfo(type)
        return

class LaneNet(ExtendedModule):
    def __init__(self, config, num_vehicles_outputs, num_lane_encoder_outputs):
        super().__init__()

        self.config = config

        self._initProps(num_vehicles_outputs, num_lane_encoder_outputs)
        self._makeNetwork()
        return
    
    def _initProps(self, num_vehicles_outputs, num_lane_encoder_outputs):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['hidden_layers']['number']['lane']

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']['lane']

        # get num_features
        num_features = num_vehicles_outputs + num_lane_encoder_outputs

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
    
    def _showInfo(self, type):
        if type == 'gradient':
            print(f"lane network:")

            counter = 1
            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                print(f"linear layer {counter}: weight = {layer.weight.grad.norm().item():.3f}, bias = {layer.bias.grad.norm().item():.3f}")
                counter += 1

        elif type == 'parameters':
            num_params = sum(p.numel() for p in self.parameters())
            print(f"lane network parameters: {num_params}")
        
        else:
            raise NotImplementedError(f"Not supported type: {type}")

        return

    def forward(self, x):
        return self.net(x)
    
    def showInfo(self, type):
        self._showInfo(type)
        return  
    
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

        self.num_hidden_layers = drl_info['architecture']['proto']['hidden_layers']['number']['lanes']

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']['lanes']

        # set num_lanes
        self.num_lanes = num_lanes

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
    
    def _showInfo(self, type):
        if type == 'gradient':
            print(f"lanes network (num lanes = {self.num_lanes}):")

            counter = 1
            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                print(f"linear layer {counter}: weight = {layer.weight.grad.norm().item():.3f}, bias = {layer.bias.grad.norm().item():.3f}")
                counter += 1
        elif type == 'parameters':
            num_params = sum(p.numel() for p in self.parameters())
            print(f"lanes network parameters (num lanes = {self.num_lanes}): {num_params}")
        
        else:
            raise NotImplementedError(f"Not supported type: {type}")

        return
    
    def forward(self, x):
        return self.net(x)
    
    def showInfo(self, type):
        self._showInfo(type)
        return
    
class RoadEncoderNet(ExtendedModule):
    def __init__(self, config, num_roads):
        super().__init__()

        self.config = config

        self._initProps(num_roads)
        self._makeNetwork()
        return
    
    def _initProps(self, num_roads):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['hidden_layers']['number']['road_encoder']

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']['road_encoder']

        # get num_features
        num_features = self.config.get('num_features_map')['road'][num_roads]

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
    
    def _showInfo(self, type):
        if type == 'gradient':
            print(f"road encoder network:")

            counter = 1
            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                print(f"linear layer {counter}: weight = {layer.weight.grad.norm().item():.3f}, bias = {layer.bias.grad.norm().item():.3f}")
                counter += 1

        elif type == 'parameters':
            num_params = sum(p.numel() for p in self.parameters())
            print(f"road encoder network parameters: {num_params}")
        
        else:
            raise NotImplementedError(f"Not supported type: {type}")

        return
    
    def forward(self, x):
        return self.net(x)
    
    def showInfo(self, type):
        self._showInfo(type)
        return


class RoadNet(ExtendedModule):
    def __init__(self, config, num_lanes_outputs, num_lanes, num_road_encoder_outputs):
        super().__init__()

        self.config = config

        self._initProps(num_lanes, num_lanes_outputs, num_road_encoder_outputs)
        self._makeNetwork()
        return

    def _initProps(self, num_lanes, num_lanes_outputs, num_road_encoder_outputs):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['hidden_layers']['number']['road']

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']['road']

        # set num_lanes
        self.num_lanes = num_lanes

        # get num_features
        num_features = num_lanes_outputs + num_road_encoder_outputs

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
    
    def _showInfo(self, type):
        if type == 'gradient':
            print(f"road network (num lanes = {self.num_lanes}):")

            counter = 1
            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                print(f"linear layer {counter}: weight = {layer.weight.grad.norm().item():.3f}, bias = {layer.bias.grad.norm().item():.3f}")
                counter += 1
        elif type == 'parameters':
            num_params = sum(p.numel() for p in self.parameters())
            print(f"road network parameters (num lanes = {self.num_lanes}): {num_params}")
        
        else:
            raise NotImplementedError(f"Not supported type: {type}")

        return
    
    def forward(self, x):
        return self.net(x)
    
    def showInfo(self, type):
        self._showInfo(type)
        return
    
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

        self.num_hidden_layers = drl_info['architecture']['proto']['hidden_layers']['number']['roads']

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']['roads']

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
    
    def _showInfo(self, type):
        if type == 'gradient':
            print(f"roads network:")

            counter = 1
            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                print(f"linear layer {counter}: weight = {layer.weight.grad.norm().item():.3f}, bias = {layer.bias.grad.norm().item():.3f}")
                counter += 1

        elif type == 'parameters':
            num_params = sum(p.numel() for p in self.parameters())
            print(f"roads network parameters: {num_params}")
        
        else:
            raise NotImplementedError(f"Not supported type: {type}")

        return
    
    def forward(self, x):
        return self.net(x)
    
    def showInfo(self, type):
        self._showInfo(type)
        return

class IntersectionEncoderNet(ExtendedModule):
    def __init__(self, config, num_roads):
        super().__init__()

        self.config = config

        self._initProps(num_roads)
        self._makeNetwork()
        return
    
    def _initProps(self, num_roads):
        # set network parameters
        drl_info = self.config.get('drl_info')

        self.num_hidden_layers = drl_info['architecture']['proto']['hidden_layers']['number']['intersection_encoder']

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']['intersection_encoder']

        # get num_features
        num_features = self.config.get('num_features_map')['intersection'][num_roads]

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
    
    def _showInfo(self, type):
        if type == 'gradient':
            print(f"intersection encoder network:")

            counter = 1
            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                print(f"linear layer {counter}: weight = {layer.weight.grad.norm().item():.3f}, bias = {layer.bias.grad.norm().item():.3f}")
                counter += 1
        
        elif type == 'parameters':
            num_params = sum(p.numel() for p in self.parameters())
            print(f"intersection encoder network parameters: {num_params}")
        
        else:
            raise NotImplementedError(f"Not supported type: {type}")

        return
    
    def forward(self, x):
        return self.net(x)

    def showInfo(self, type):
        self._showInfo(type)
        return
    
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

        self.num_hidden_layers = drl_info['architecture']['proto']['hidden_layers']['number']['intersection']

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']['intersection']

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
    
    def _showInfo(self, type):
        if type == 'gradient':
            print(f"intersection network:")

            counter = 1
            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                print(f"linear layer {counter}: weight = {layer.weight.grad.norm().item():.3f}, bias = {layer.bias.grad.norm().item():.3f}")
                counter += 1
        
        elif type == 'parameters':
            num_params = sum(p.numel() for p in self.parameters())
            print(f"intersection network parameters: {num_params}")

        else:
            raise NotImplementedError(f"Not supported type: {type}")

        return
    
    def forward(self, x):
        return self.net(x)
    
    def showInfo(self, type):
        self._showInfo(type)
        return
    
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

        self.num_hidden_layers = drl_info['architecture']['proto']['hidden_layers']['number']['dueling']

        self.activation_function = drl_info['architecture']['common']['activation_function']['type']
        if self.activation_function == 'leaky_relu':
            self.alpha = drl_info['architecture']['common']['activation_function']['leaky_relu']['alpha']
        
        self.compression_type = drl_info['architecture']['proto']['compression']['type']
        self.compression_rate = drl_info['architecture']['proto']['compression']['rate']['dueling']

        # set num_input, num_output
        self.num_inputs = num_intersection_outputs
        self.num_outputs_map = {
            'value': 1,
            'advantage': self.config.get('num_features_map')['intersection'][num_roads]
        } 

        # set num_hidden_layer_features_map
        self.num_hidden_layer_features_map = {'value': {}, 'advantage': {}}
        for part in ['value', 'advantage']:
            for layer_id in range(1, self.num_hidden_layers + 1):
                if self.compression_type == 'linear':
                    self.num_hidden_layer_features_map[part][layer_id] = int(self.num_inputs - (layer_id * (self.num_inputs - self.num_inputs * self.compression_rate) / (self.num_hidden_layers + 1)))
                elif self.compression_type == 'geometric':
                    self.num_hidden_layer_features_map[part][layer_id] = int(self.num_inputs * (self.num_inputs * self.compression_rate / self.num_inputs) ** (layer_id / (self.num_hidden_layers + 1)))
                else:
                    raise NotImplementedError(f"Not supported compression type: {self.compression_type}")

        return
    
    def _makeNetwork(self):
        self.network_map = nn.ModuleDict()

        if self.num_hidden_layers == 0:
            for part in ['value', 'advantage']:
                self.network_map[part] = nn.Sequential(nn.Linear(self.num_inputs, self.num_outputs_map[part]))
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
    
    def _showInfo(self, type):
        if type == 'gradient':
            for part in ['value', 'advantage']:
                print(f"{part} network:")
                counter = 1
                for layer in self.network_map[part]:
                    if not isinstance(layer, nn.Linear):
                        continue

                    print(f"linear layer {counter}: weight = {layer.weight.grad.norm().item():.3f}, bias = {layer.bias.grad.norm().item():.3f}")
                    counter += 1

        elif type == 'parameters':
            for part in ['value', 'advantage']:
                num_params = sum(p.numel() for p in self.network_map[part].parameters())
                print(f"{part} network parameters: {num_params}")

        else:
            raise NotImplementedError(f"Not supported type: {type}")
    
    def forward(self, x):
        return self.network_map['value'](x), self.network_map['advantage'](x)
    
    def showInfo(self, type):
        self._showInfo(type)
        return
