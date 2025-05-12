from config.config import Config
from objects.vissim import Vissim

config = Config()

vissim = Vissim(config)

vissim.run()

vissim.exit()

print('Stopper')