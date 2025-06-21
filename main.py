from config.config import Config
from libs.executor import Executor
from objects.vissim import Vissim

# 設定オブジェクトと非同期オブジェクトを初期化
config = Config()
executor = Executor(config)

# シミュレーションを実行
simulator_info = config.get('simulator_info')
for count in range(simulator_info['simulation_count']):
    vissim = Vissim(config, executor)

    vissim.run()
    vissim.exit()
    
    print(f"Simulation {count + 1} completed.")

# 非同期オブジェクトをシャットダウン
executor.shutdown()
