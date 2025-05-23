import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def get(self, property_name):
        if hasattr(self, property_name) == False:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{property_name}'")
        
        return getattr(self, property_name)
        
    def set(self, property_name, value):
        setattr(self, property_name, value)