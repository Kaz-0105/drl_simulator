from objects.vissim import Vissim
from pathlib import Path

def main():
    vissim = Vissim(Path(__file__).parent)
    vissim.run()
    return

if __name__ == "__main__":
    main()