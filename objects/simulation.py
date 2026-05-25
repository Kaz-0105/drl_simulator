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
        self.network = None # set in network object initialization

        self._initProps()
        return
    
    @property
    def finish_flg(self):
        if self.control_method == 'drl':
            if self.current_time < self.simulation_time:
                return False
            
            if self.network.get('drl_framework') == 'apex':
                return self.network.master_agents.get('finish_flg')
            
            else:
                raise NotImplementedError(f"Not supported DRL framework: {self.network.get('drl_framework')}")

        elif self.control_method in ['mpc', 'scoot']:
            return self.current_time >= self.simulation_time

        else:
            raise NotImplementedError(f"Not supported control method: {self.control_method}")
    
    def _initProps(self):
        # set simulator information
        simulator_info = self.config.get('simulator_info')
        self.control_method = simulator_info['control_method']
        self.layout_name = simulator_info['layout_name']
        self.inflow_name = simulator_info['inflow_name']

        self.simulation_time = simulator_info['simulation_time']
        self.time_step = simulator_info['time_step']
        
        self.debug_flg = simulator_info['debug']['flg']

        if self.control_method == 'drl':
            drl_info = self.config.get('drl_info')
            self.drl_simulation_type = drl_info['simulation_type']
            self.drl_framework = drl_info['framework']['type']
        return
    
    def _updateProps(self, simulation_id):
        self.id = simulation_id

        # set current_time
        self.current_time = self.com.AttValue('SimSec')

        # set seed
        simulator_info = self.config.get('simulator_info')
        if self.control_method == 'drl':
            if self.drl_simulation_type == 'train':
                self.seed = random.randint(100 + 1, 10000)
            elif self.drl_simulation_type == 'test':
                self.seed = simulator_info['seed']
            else:
                raise NotImplementedError(f"Not supported DRL simulation type: {self.drl_simulation_type}")

        elif self.control_method in ['mpc', 'scoot']:
            self.seed = simulator_info['seed']

        else:
            raise NotImplementedError(f"Not supported control method: {self.control_method}")
        
        return
        
    def _setParametersToVissim(self):
        # Vissimにパラメータを設定
        self.com.SetAttValue('RandSeed', self.seed)
        self.com.SetAttValue('SimPeriod', self.simulation_time + 2)

        # シミュレーションの速度について
        self.com.SetAttValue('UseAllCores', True)
        self.com.SetAttValue('UseMaxSimSpeed', True)

        # Queueの測定とDelayの測定とDataCollectionの測定の設定
        evaluation_com = self.vissim.com.Evaluation
        evaluation_com.SetAttValue('DelaysCollectData', True)
        evaluation_com.SetAttValue('DelaysFromTime', 0)
        evaluation_com.SetAttValue('DelaysToTime', self.simulation_time + 2)
        evaluation_com.SetAttValue('DelaysInterval', self.time_step)

        evaluation_com.SetAttValue('QueuesCollectData', True)
        evaluation_com.SetAttValue('QueuesFromTime', 0)
        evaluation_com.SetAttValue('QueuesToTime', self.simulation_time + 2)
        evaluation_com.SetAttValue('QueuesInterval', self.time_step)

        evaluation_com.SetAttValue('DataCollCollectData', True)
        evaluation_com.SetAttValue('DataCollFromTime', 0)
        evaluation_com.SetAttValue('DataCollToTime', self.simulation_time + 2)
        evaluation_com.SetAttValue('DataCollInterval', self.time_step)
        return
    
    def _runApex(self):
        # get local agents and master agents
        local_agents = self.network.local_agents
        master_agents = self.network.master_agents

        # update network and make state for each agent
        self.network.update('initial')
        local_agents.update('initial_state')
        
        while self.current_time < self.simulation_time:
            # sync local agents
            if self.drl_simulation_type == 'train':
                local_agents.sync(type='model')

            # get action
            local_agents.update('action')

            # run single step
            self._runSingleStep()

            # update network and get state and reward
            self.network.update()
            local_agents.update('state')

            # show action and reward
            local_agents.showInfo('action_result')

            # update buffer and train network
            master_agents.update('buffer')
            master_agents.train()

            # if done flg is True, break loop
            if local_agents.get('done_flg'):
                break
        
        # final network update
        self.network.update('final')

        # show the results of this episode
        master_agents.showInfo('result')

        # save performance metrics
        self.network.save()
        return
    
    def _runMpc(self):
        # get mpc_controllers and bc_buffers
        mpc_controllers = self.network.mpc_controllers
        if self.network.get('bc_flg'):
            bc_buffers = self.network.bc_buffers

        while self.current_time < self.simulation_time:
            self.network.update()

            mpc_controllers.optimize()

            if self.network.get('bc_flg'):
                mpc_controllers.update('bc')
                bc_buffers.save()

            self._runSingleStep()
        
        if self.network.get('bc_flg'):
            bc_buffers.writeToFile()
        
        self.network.update(type='final')
        self.network.save()
        return
    
    def _runBc(self):
        # 行動クローンを行う
        bc_agent = self.network.bc_agent
        bc_agent.cloneExpert()

        while self.current_time < self.simulation_time:
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

        self.network.save()
        return
    
    def _runScoot(self):
        scoot_controllers = self.network.scoot_controllers

        while self.current_time < self.simulation_time:
            self.network.update()
            scoot_controllers.update()
            self._runSingleStep()
        
        self.network.update(type='final')
        self.network.save()

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
        return

    def _runForDebug(self):
        # 30秒進める
        self.com.SetAttValue('SimBreakAt', self.current_time + 30)
        self.com.RunContinuous()

        # 現在時刻を更新
        self.current_time += 30
        return
    
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

        if self.control_method == 'drl' and self.drl_framework == 'apex':
            self._runApex()  
        elif self.control_method == 'mpc':
            self._runMpc()
        elif self.control_method == 'bc':
            self._runBc()
        elif self.control_method == 'scoot':
            self._runScoot()
        else:
            raise NotImplementedError(f"Not supported control method: {self.control_method}")

        self._showInfo('end')
        return