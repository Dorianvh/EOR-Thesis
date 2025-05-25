from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from battery_env import BatteryEnv

print("[TRAINING] Initializing training environment...")
env = BatteryEnv("data/ID1_prices_germany.csv")
eval_env = BatteryEnv("data/ID1_prices_germany.csv")

print("[TRAINING] Creating DQN model...")
model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.95,
    target_update_interval=250,
    tensorboard_log="./tensorboard_logs/"
)

print("[TRAINING] Defining evaluation callback...")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/',
    log_path='./logs/',
    eval_freq=5000,
    deterministic=True,
    render=False
)

print("[TRAINING] Starting training...")
model.learn(total_timesteps=500, callback=eval_callback)

print("[TRAINING] Training complete. Saving model...")
model.save("dqn_battery")

print("[TRAINING] Model saved as 'dqn_battery.zip'")