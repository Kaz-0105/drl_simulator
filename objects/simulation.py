from libs.common import Common

import random

class Simulation(Common):
    RED = 1
    GREEN = 3
    
    def __init__(self, vissim):
        super().__init__()

        # set config and vissim object
        self.config = vissim.config
        self.vissim = vissim

        self._initProps()
        return
    
    @property
    def finish_flg(self):
        if self.control_method == 'drl':
            if self.current_time < self.end_time:
                return False
            
            if self.network.get('drl_framework') == 'apex':
                return self.network.master_agents.get('finish_flg')
            
            else:
                raise NotImplementedError(f"Not supported DRL framework: {self.network.get('drl_framework')}")

        elif self.control_method in ['mpc', 'scoot']:
            return self.current_time >= self.end_time

        else:
            raise NotImplementedError(f"Not supported control method: {self.control_method}")
    
    def _initProps(self):
        # set simulator information
        simulator_info = self.config.get('simulator_info')
        self.control_method = simulator_info['control_method']
        self.layout_name = simulator_info['layout_name']
        self.inflow_name = simulator_info['inflow_name']

        self.end_time = simulator_info['simulation_time']
        self.time_step = simulator_info['time_step']
        
        self.debug_flg = simulator_info['debug']['flg']
        return
    
    def _updateProps(self, simulation_id):
        self.id = simulation_id

        # set current_time
        self.current_time = self.com.AttValue('SimSec')

        # set seed
        simulator_info = self.config.get('simulator_info')
        if simulator_info['seed']['is_random']:
            self.seed = random.randint(100 + 1, 10000)
        else:
            self.seed = simulator_info['seed']['value']
        return
        
    def _setParametersToVissim(self):
        # Vissimにパラメータを設定
        self.com.SetAttValue('RandSeed', self.seed)
        self.com.SetAttValue('SimPeriod', self.end_time + 2)

        # シミュレーションの速度について
        self.com.SetAttValue('UseAllCores', True)
        self.com.SetAttValue('UseMaxSimSpeed', True)

        # Queueの測定とDelayの測定とDataCollectionの測定の設定
        evaluation_com = self.vissim.com.Evaluation
        evaluation_com.SetAttValue('DelaysCollectData', True)
        evaluation_com.SetAttValue('DelaysFromTime', 0)
        evaluation_com.SetAttValue('DelaysToTime', self.end_time + 2)
        evaluation_com.SetAttValue('DelaysInterval', self.time_step)

        evaluation_com.SetAttValue('QueuesCollectData', True)
        evaluation_com.SetAttValue('QueuesFromTime', 0)
        evaluation_com.SetAttValue('QueuesToTime', self.end_time + 2)
        evaluation_com.SetAttValue('QueuesInterval', self.time_step)

        evaluation_com.SetAttValue('DataCollCollectData', True)
        evaluation_com.SetAttValue('DataCollFromTime', 0)
        evaluation_com.SetAttValue('DataCollToTime', self.end_time + 2)
        evaluation_com.SetAttValue('DataCollInterval', self.time_step)
        return
    
    def activate(self, simulation_id):
        # set com object
        self.com = self.vissim.com.Simulation

        self._updateProps(simulation_id)
        self._setParametersToVissim()
        return

    def run(self):
        self._showInfo('start')
        
        if self.debug_flg:
            self._runForDebug()
        
        self._getSignalControlAuth()

        if self.control_method == 'drl':
            local_agents = self.network.local_agents
            master_agents = self.network.master_agents

            self.network.update(type='initial')
            local_agents.update(type='initial_state')
            
            while self.current_time < self.end_time:
                # get action
                local_agents.update(type='action')

                # run single step
                self._runSingleStep()

                # update network and get state and reward
                self.network.update()
                local_agents.update(type='state')

                # update buffer and train network
                master_agents.updateBuffer()
                master_agents.train()

                # if done flg is True, break loop
                if local_agents.get('done_flg'):
                    break
            
            # final network update
            self.network.update(type='final')

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
                self.network.update()

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
            self.network.update(type='final')
        
        elif self.control_method == 'bc':
            # 行動クローンを行う
            bc_agent = self.network.bc_agent
            bc_agent.cloneExpert()

            while self.current_time < self.end_time:
                # 最初のネットワークの更新
                self.network.update(type='initial')

                # 状態・報酬・行動を計算
                bc_agent.updateState()
                bc_agent.updateReward()
                bc_agent.updateAction()

                # Vissimを1ステップ進める
                self._runSingleStep()
            
            # 最後のネットワーク更新
            self.network.update(type='final')

            # トータルの報酬を表示し、モデルを保存
            bc_agent.showTotalReward()
            bc_agent.saveModel()
        
        elif self.control_method == 'scoot':
            scoot_controllers = self.network.scoot_controllers

            while self.current_time < self.end_time:
                self.network.update()
                scoot_controllers.updateParameters()
                self._runSingleStep()
            
            self.network.update(type='final')

        # save performance metrics
        self.network.save()

        self._showInfo('end')
        return
    
    def _showInfo(self, type):
        print('==============================================')

        if type == 'start':
            print('status: start simulation')
            if self.control_method == 'drl':
                print(f"simulation id: {self.id}")
            print(f"control method: {self.control_method}")

        elif type == 'end':
            print('status: end simulation')
            print(f"simulation id: {self.id}")
        
        else:
            raise NotImplementedError(f"Not supported type: {type}")

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
                signal_group.com.SetAttValue('SigState', self.RED)

    @property
    def network(self):
        return self.vissim.network