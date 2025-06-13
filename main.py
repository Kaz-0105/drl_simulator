from config.config import Config
from objects.vissim import Vissim

config = Config()

for count in range(1000):
    vissim = Vissim(config)

    vissim.run()

    vissim.exit()
    
    print(f"Run {count + 1} completed.")