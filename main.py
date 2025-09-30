from config.config import Config
from libs.executor import Executor
from objects.vissim import Vissim
from libs.shared_resource import SharedResources

# 設定オブジェクトと非同期オブジェクトを初期化
config = Config()
executor = Executor(config)

# 引継ぎ用のオブジェクトを初期化
shared_resources = SharedResources()

# シミュレーションを実行
simulator_info = config.get('simulator_info')
for sim_count in range(1, simulator_info['num_simulations'] + 1):
    # vissimオブジェクトを初期化
    vissim = Vissim(config, executor, shared_resources, sim_count)

    # シミュレーションを起動
    vissim.run()
    vissim.exit()
    vissim.backup()
    
    # 終了したことを通知
    print(f"Simulation {sim_count} completed.")

    # 終了フラグが立っていたら終了
    if vissim.get('finish_flg'):
        break

# 非同期オブジェクトをシャットダウン
executor.shutdown()
