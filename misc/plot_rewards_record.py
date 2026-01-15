from pathlib import Path
import pickle
import matplotlib.pyplot as plt



# 見たいモデルを選択
model_id = 12

# 保存する結果
fig_flgs = {
    'total_reward_record': True,
    'epsilon_record': True,
    'num_epochs_record': True,
    'update_interval_record': True,
    'random_phase_probs_record': True,
}

# 指数移動平均を使うか
ema_flg = True

# 結果保存先パスを取得
results_path = (Path(__file__).parent/ '..' / 'results').resolve()

# create plot directory if not exists
plot_dir = results_path / 'plots'
plot_dir.mkdir(parents=True, exist_ok=True)

# ファイル名を指定してデータを読み込む
session_dir_path = results_path / 'session' / 'drl' / 'apex'/ f'session_{model_id}' / 'session.pkl'
if not session_dir_path.exists():
    raise FileNotFoundError(f"Session file {session_dir_path} does not exist.")

with open(session_dir_path, 'rb') as f:
    session_data = pickle.load(f)
    session_data['episode_record'] = list(range(1, len(session_data['total_reward_record']) + 1))

if fig_flgs['total_reward_record']:
    # total_rewardの推移を描画
    alpha = 0.1 if ema_flg else 1.0
    ema = 0
    
    total_reward_record = []
    for idx, total_reward in enumerate(session_data['total_reward_record']):
        ema = alpha * total_reward + (1 - alpha) * ema if idx > 0 else total_reward
        total_reward_record.append(ema)
    
    simulation_time_record = session_data['simulation_time_record']
    for idx in range(len(total_reward_record)):
        total_reward_record[idx] *= 500 / simulation_time_record[idx]
        session_data['total_reward_record'] = total_reward_record

    fig_total_reward, ax_total_reward = plt.subplots(figsize=(16,9))
    ax_total_reward.plot(session_data['episode_record'], session_data['total_reward_record'], linewidth=4, label=f'Model {model_id}')
    ax_total_reward.set_xlabel('Episode', fontsize=20)
    ax_total_reward.set_ylabel('Total Reward', fontsize=20)
    ax_total_reward.set_title('Total Reward Record (EMA)' if ema_flg else 'Total Reward Record', fontsize=24)
    ax_total_reward.tick_params(axis='both', which='major', labelsize=20)
    ax_total_reward.tick_params(axis='both', which='minor', labelsize=20)
    ax_total_reward.legend(fontsize=16)

    fig_total_reward.savefig(results_path / 'plots' / f"total_reward_record_{'ema' if ema_flg else 'raw'}.png")

if fig_flgs['epsilon_record']:
    # epsilonの推移を描画
    fig_epsilon, ax_epsilon = plt.subplots(figsize=(16,9))

    ax_epsilon.plot(session_data['episode_record'], session_data['epsilon_record'], linewidth=4, label=f'Model {model_id}')
    ax_epsilon.set_xlabel('Episode', fontsize=20)
    ax_epsilon.set_ylabel('Epsilon', fontsize=20)
    ax_epsilon.set_title('Epsilon Record', fontsize=24)
    ax_epsilon.tick_params(axis='both', which='major', labelsize=20)
    ax_epsilon.tick_params(axis='both', which='minor', labelsize=20)
    ax_epsilon.legend(fontsize=16)

    fig_epsilon.savefig(results_path / 'plots' / "epsilon_record.png")

if fig_flgs['num_epochs_record']:
    # num_epochsの推移を描画
    fig_num_epochs, ax_num_epochs = plt.subplots(figsize=(16,9))

    ax_num_epochs.plot(session_data['episode_record'], session_data['num_epochs_record'], linewidth=4, label=f'Model {model_id}')
    ax_num_epochs.set_xlabel('Episode', fontsize=20)
    ax_num_epochs.set_ylabel('Num Epochs', fontsize=20)
    ax_num_epochs.set_title('Num Epochs Record', fontsize=24)
    ax_num_epochs.tick_params(axis='both', which='major', labelsize=20)
    ax_num_epochs.tick_params(axis='both', which='minor', labelsize=20)
    ax_num_epochs.legend(fontsize=16)

    fig_num_epochs.savefig(results_path / 'plots' / "num_epochs_record.png")

if fig_flgs['update_interval_record']:
    # update_intervalの推移を描画
    fig_update_interval, ax_update_interval = plt.subplots(figsize=(16,9))
    
    ax_update_interval.plot(session_data['episode_record'], session_data['update_interval_record'], linewidth=4, label=f'Model {model_id}')
    ax_update_interval.set_xlabel('Episode', fontsize=20)
    ax_update_interval.set_ylabel('Update Interval', fontsize=20)
    ax_update_interval.set_title('Update Interval Record', fontsize=24)
    ax_update_interval.tick_params(axis='both', which='major', labelsize=20)
    ax_update_interval.tick_params(axis='both', which='minor', labelsize=20)
    ax_update_interval.legend(fontsize=16)

    fig_update_interval.savefig(results_path / 'plots' / "update_interval_record.png")


if fig_flgs['random_phase_probs_record']:
    fig_random_phase_probs, ax_random_phase_probs = plt.subplots(figsize=(16,9))

    for phase_id in range(1, len(session_data['random_phase_probs_record']) + 1):
        tmp_phase_probs_record = session_data['random_phase_probs_record'][phase_id]
        ax_random_phase_probs.plot(session_data['episode_record'], tmp_phase_probs_record, linewidth=4, label=f'Phase {phase_id}')
    
    ax_random_phase_probs.set_xlabel('Episode', fontsize=20)
    ax_random_phase_probs.set_ylabel('Random Phase Probability', fontsize=20)
    ax_random_phase_probs.set_title('Random Phase Probabilities Record', fontsize=24)
    ax_random_phase_probs.tick_params(axis='both', which='major', labelsize=20)
    ax_random_phase_probs.tick_params(axis='both', which='minor', labelsize=20)
    ax_random_phase_probs.legend(fontsize=16)

    fig_random_phase_probs.savefig(results_path / 'plots' / "random_phase_probs_record.png")

