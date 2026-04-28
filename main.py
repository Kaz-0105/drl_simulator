from objects.vissim import Vissim
from pathlib import Path

vissim = Vissim(Path(__file__).parent)
vissim.run()