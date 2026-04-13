import torch 
import copy

class ExtendedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        return
    
    def get(self, property_name, type='reference'):
        if not self.has(property_name):
            raise AttributeError(f"Not found property '{property_name}' in {self.__class__.__name__}.")
        
        if type == 'reference':
            return getattr(self, property_name)
        elif type == 'copy':
            return copy.deepcopy(getattr(self, property_name))
        else:
            raise NotImplementedError(f"Not supported type: {type}")
    
    def set(self, property_name, value):
        setattr(self, property_name, value)
        return
    
    def has(self, property_name):
        return hasattr(self, property_name)
    
    def getPropertyNames(self):
        return list(self.__dict__.keys())
    
    def toDevice(self, data, device):
        if isinstance(data, dict):
            return {key: self.toDevice(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.toDevice(item, device) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data
    
class ExtendedDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        return
    
    def get(self, property_name, type='reference'):
        if not self.has(property_name):
            raise AttributeError(f"Not found property '{property_name}' in {self.__class__.__name__}.")
        
        if type == 'reference':
            return getattr(self, property_name)
        elif type == 'copy':
            return copy.deepcopy(getattr(self, property_name))
        else:
            raise NotImplementedError(f"Not supported type: {type}")
    
    def set(self, property_name, value):
        setattr(self, property_name, value)
        return
    
    def has(self, property_name):
        return hasattr(self, property_name)
    
    def getPropertyNames(self):
        return list(self.__dict__.keys())
    
    def toDevice(self, data, device):
        if isinstance(data, dict):
            return {key: self.toDevice(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.toDevice(item, device) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data