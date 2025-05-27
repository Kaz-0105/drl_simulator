from libs.common import Common

class Simulation(Common):
    def __init__(self, vissim):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = vissim.config
        self.vissim = vissim

        # comオブジェクトを取得
        self.com = self.vissim.com.Simulation

        # 現在時刻と終了時刻を取得
        self.current_time = self.com.AttValue('SimSec')
        simulation_info = self.config.get('simulator_info')
        self.end_time = simulation_info['simulation_time']

        # シード値を取得
        self.random_seed = simulation_info['random_seed']

        # タイムステップを取得
        self.time_step = simulation_info['time_step']

        # vissimに反映
        self.setParametersToVissim()

    def setParametersToVissim(self):
        self.com.SetAttValue('RandSeed', self.random_seed)
        self.com.SetAttValue('SimPeriod', self.end_time + 1) # Vissimの仕様上、終了時刻に達するとネットワークの情報が消えるので１秒長くして消えないようにする

    def run(self):
        simulation_info = self.config.get('simulator_info')
        if simulation_info['control_method'] == 'drl':
            # デバッグ用
            self.runForDebug()

            # 最初のネットワークの更新
            self.network.updateData()

            # 状態量を計算
            self.network.local_agents.getState()
            
            while self.current_time < self.end_time:
                # 行動を計算
                self.network.local_agents.getAction()

                # Vissimを1ステップ進める
                self.runSingleStep()

                # ネットワークの更新
                self.network.updateData()

                # 報酬を計算
                self.network.local_agents.getReward()

                # 次の状態量を計算
                self.network.local_agents.getState()

                # バッファーに送るデータを作成
                self.network.local_agents.makeLearningData()

                # データをバッファーに保存
                self.network.master_agents.saveLearningData()


    def runSingleStep(self):
        # タイムステップ分進める
        self.com.SetAttValue('SimBreakAt', self.current_time + self.time_step)
        self.com.RunContinuous()

        # 現在時刻を更新
        self.current_time += self.time_step

    def runForDebug(self):
        # 30秒進める
        self.com.SetAttValue('SimBreakAt', self.current_time + 30)
        self.com.RunContinuous()

        # 現在時刻を更新
        self.current_time += 30