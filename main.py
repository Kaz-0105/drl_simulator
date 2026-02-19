from objects.vissim import Vissim
from pathlib import Path
from misc.clean.utils import clean_performance_metric_dir, clean_layout_dir

vissim = Vissim(Path(__file__).parent)
vissim.run()

clean_performance_metric_dir()
clean_layout_dir()



