from config.config import Config
from objects.vissim import Vissim

config = Config()

for _ in range(5):
    vissim = Vissim(config)

    vissim.run()

    vissim.exit()