from config.config import Config
from objects.vissim import Vissim

config = Config()

simulator_info = config.get('simulator_info')
for count in range(simulator_info['simulation_count']):
    vissim = Vissim(config)

    vissim.run()

    vissim.exit()
    
    print(f"Simulation {count + 1} completed.")