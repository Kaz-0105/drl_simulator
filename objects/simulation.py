from libs.common import Common

import random

class Simulation(Common):
    def __init__(self, vissim):
        # 継承
        super().__init__()

        # 設定オブジェクトと上位の紐づくオブジェクトを取得
        self.config = vissim.config
        self.vissim = vissim

        # comオブジェクトを取得
        self.com = self.vissim.com.Simulation

        # 現在時刻を取得
        self.current_time = self.com.AttValue('SimSec')

        # 各種設定情報を取得
        simulation_info = self.config.get('simulator_info')
        self.end_time = simulation_info['simulation_time']
        self.time_step = simulation_info['time_step']
        self.control_method = simulation_info['control_method']
        self.debug_flg = simulation_info['debug']['flg']

        # シード値を設定
        self._makeRandomSeed(simulation_info)

        # vissimに反映
        self._setParametersToVissim()

    def _makeRandomSeed(self, simulation_info):
        seed_info = simulation_info['seed']

        if seed_info['is_random']:
            self.random_seed = random.randint(100 + 1, 10000)
        else:
            self.random_seed = seed_info['value']
        
    def _setParametersToVissim(self):
        # Vissimにパラメータを設定
        self.com.SetAttValue('RandSeed', self.random_seed)
        self.com.SetAttValue('SimPeriod', self.end_time + 1)

        # シミュレーションの速度について
        self.com.SetAttValue('UseAllCores', True)
        self.com.SetAttValue('UseMaxSimSpeed', True)

        # Queueの測定とDelayの測定とDataCollectionの測定の設定
        evaluation_com = self.vissim.com.Evaluation
        evaluation_com.SetAttValue('DelaysCollectData', True)
        evaluation_com.SetAttValue('DelaysFromTime', 0)
        evaluation_com.SetAttValue('DelaysToTime', self.end_time + 1)
        evaluation_com.SetAttValue('DelaysInterval', self.time_step)

        evaluation_com.SetAttValue('QueuesCollectData', True)
        evaluation_com.SetAttValue('QueuesFromTime', 0)
        evaluation_com.SetAttValue('QueuesToTime', self.end_time + 1)
        evaluation_com.SetAttValue('QueuesInterval', self.time_step)

        evaluation_com.SetAttValue('DataCollCollectData', True)
        evaluation_com.SetAttValue('DataCollFromTime', 0)
        evaluation_com.SetAttValue('DataCollToTime', self.end_time + 1)
        evaluation_com.SetAttValue('DataCollInterval', self.time_step)
        return

    def run(self):
        # デバックフラグが立っているとき30秒進める
        if self.debug_flg:
            self._runForDebug()
        
        # 信号機の操作権限をこちら側に移す
        self._getSignalControlAuth()

        if self.control_method == 'drl':
            # 必要なオブジェクトを取得
            local_agents = self.network.local_agents
            master_agents = self.network.master_agents

            # 最初のネットワーク更新と状態の取得
            self.network.updateData()
            local_agents.getState()
            
            while self.current_time < self.end_time:
                # 行動を取得
                local_agents.getAction()

                # Vissimを1ステップ進める
                self._runSingleStep()

                # ネットワークの更新，状態・報酬の取得，学習データの作成
                self.network.updateData()
                local_agents.getState()
                local_agents.getReward()
                local_agents.makeLearningData()

                # データを保存し学習
                master_agents.saveLearningData()
                master_agents.train()

                # 終了フラグが立っていた場合終了
                if local_agents.get('done_flg'):
                    break
            
            # 最後のネットワーク更新
            self.network.updateData()

            # トータルの報酬を更新し，データを保存
            master_agents.updateSessionData()
            master_agents.save()
            
        elif self.control_method == 'mpc':
            # 必要なオブジェクトを取得
            mpc_controllers = self.network.mpc_controllers
            if self.network.get('bc_flg'):
                bc_buffers = self.network.bc_buffers

            while self.current_time < self.end_time:
                # ネットワークの更新
                self.network.updateData()

                # MPCで最適な行動を計算
                mpc_controllers.optimize()

                # 行動クローン用のデータを作成
                if self.network.get('bc_flg'):
                    mpc_controllers.updateBcData()
                    bc_buffers.saveBcData()

                # Vissimを1ステップ進める
                self._runSingleStep()
            
            # bcバッファのデータをファイルに保存
            if self.network.get('bc_flg'):
                bc_buffers.writeToFile()
            
            # 最後のネットワーク更新
            self.network.updateData()
        
        elif self.control_method == 'bc':
            # 行動クローンを行う
            bc_agent = self.network.bc_agent
            bc_agent.cloneExpert()

            while self.current_time < self.end_time:
                # 最初のネットワークの更新
                self.network.updateData()

                # 状態・報酬・行動を計算
                bc_agent.updateState()
                bc_agent.updateReward()
                bc_agent.updateAction()

                # Vissimを1ステップ進める
                self._runSingleStep()
            
            # 最後のネットワーク更新
            self.network.updateData()

            # トータルの報酬を表示し、モデルを保存
            bc_agent.showTotalReward()
            bc_agent.saveModel()
        
        elif self.control_method == 'scoot':
            scoot_controllers = self.network.scoot_controllers

            while self.current_time < self.end_time:
                self.network.updateData()
                scoot_controllers.updateParameters()
                self._runSingleStep()
            
            self.network.updateData()

        # 評価指標の保存
        self.network.saveData()
        return

    def _runSingleStep(self):
        # 信号現示を更新する
        self.network.signal_controllers.setNextPhaseToVissim()

        # タイムステップ分進める
        self.com.SetAttValue('SimBreakAt', self.current_time + self.time_step)
        self.com.RunContinuous()

        # 現在時刻を更新
        self.current_time += self.time_step

    def _runForDebug(self):
        # 30秒進める
        self.com.SetAttValue('SimBreakAt', self.current_time + 30)
        self.com.RunContinuous()

        # 現在時刻を更新
        self.current_time += 30
    
    def _getSignalControlAuth(self):
        # 1秒進める
        self.com.SetAttValue('SimBreakAt', self.current_time + 1)
        self.com.RunContinuous()

        # 現在時刻を更新
        self.current_time += 1

        # 信号機の操作権限を取得
        for signal_controller in self.network.signal_controllers.getAll():
            for signal_group in signal_controller.signal_groups.getAll():
                signal_group.com.SetAttValue('SigState', 1)