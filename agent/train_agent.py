import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from battery_env import BatteryEnv

print("[TRAINING] Initializing training environment...")

env_df = pd.read_csv("../data/preprocessed_data_2023.csv")

env = BatteryEnv(env_df)
eval_env = BatteryEnv(env_df)

print("[TRAINING] Creating DQN model...")
model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=250,
    tensorboard_log="./tensorboard_logs/",
    exploration_fraction=0.1,  # Increase exploration duration
    exploration_initial_eps=1.0,  # Start with full exploration
    exploration_final_eps=0.01  # Ensure some exploration remains
)

print("[TRAINING] Defining evaluation callback...")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/',
    log_path='logs/',
    eval_freq=10000,
    n_eval_episodes=1,
    deterministic=True,
    render=False
)

print("[TRAINING] Starting training...")
model.learn(total_timesteps=1000000, callback=eval_callback)

print("[TRAINING] Training complete. Saving model...")
model.save("./models/dqn_battery")

print("[TRAINING] Model saved as 'dqn_battery.zip'")