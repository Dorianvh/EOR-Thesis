import pandas as pd
import multiprocessing
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from battery_env import BatteryEnv

def train():
    print("[TRAINING] Initializing training environment...")
    env_df = pd.read_csv("../data/preprocessed_data_2023.csv")

    # Create a vectorized environment with multiple parallel environments
    num_envs = 14  # Adjust based on your CPU cores
    env = make_vec_env(
        lambda: BatteryEnv(env_df),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv  # Use SubprocVecEnv for true parallelism
    )

    # For evaluation, we can use a single environment
    eval_env = make_vec_env(lambda: BatteryEnv(env_df), n_envs=1)

    print("[TRAINING] Creating DQN model...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size= int(100000 * (num_envs ** 0.5)),  # ~374,000 with 14 envs
        learning_starts=1000 * num_envs,  # Adjust learning starts for vectorized environment
        batch_size=64 ,
        gamma=0.99, # High gamma well suited for long-term tasks like battery management
        target_update_interval=int(250 * (num_envs ** 0.5)),  # ~935 with 14 envs
        gradient_steps=num_envs,  # This makes updates scale with collection speed
        tensorboard_log="./tensorboard_logs/",
        exploration_fraction=0.1,  # Fraction of total training to decay exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01
    )

    print("[TRAINING] Defining evaluation callback...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='logs/',
        eval_freq=10000 // num_envs,  # Adjust for vectorized environment
        n_eval_episodes=1,
        deterministic=True,
        render=False
    )

    print("[TRAINING] Starting training...")
    model.learn(total_timesteps=1000000, callback=eval_callback)

    print("[TRAINING] Training complete. Saving model...")
    model.save("./models/dqn_battery")

    print("[TRAINING] Model saved as 'dqn_battery_49.zip'")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Needed for Windows
    train()