import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import os

from battery_env import BatteryEnv


def run_episode(env, model: DQN):
    """
    Runs one episode in `env` using `model`, returns a list of actions and rewards.
    """
    obs = env.reset()  # Vectorized env reset only returns observation
    done = False

    actions, rewards = [], []
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)  # Vectorized env step returns 4 values

        # For vectorized env with a single env, done is a list/array
        done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones

        # Extract scalar values from numpy arrays if needed
        if isinstance(action, np.ndarray) and action.size == 1:
            action = action.item()
        if isinstance(reward, np.ndarray) and reward.size == 1:
            reward = reward.item()

        actions.append(action)
        rewards.append(reward)

    return actions, rewards, info


def main():
    # 1) Load trained model
    model_dir = "training_runs/final/run_7"
    model_path = f"{model_dir}/best_model.zip"
    norm_path = f"{model_dir}/vecnormalize.pkl"
    print(f"Loading model from {model_path}")

    # 2) Use a different dataset for inference
    data_path = "../data/preprocessed_data_2023.csv"
    print(f"Loading inference data from {data_path}")
    df = pd.read_csv(data_path)

    feature_cols = ['ID1_price', 'Hour', 'DayOfWeek', 'Month', 'ARMAX_forecast_6hour', 'ARMAX_forecast_12hour', 'ARMAX_forecast_24hour']
    # 3) Create environment with the new data
    env = BatteryEnv(df, feature_cols)

    # Wrap the environment to match what was used during training
    env = DummyVecEnv([lambda: env])

    # 4) Load the normalization statistics
    if os.path.exists(norm_path):
        print(f"Loading normalization from {norm_path}")
        env = VecNormalize.load(norm_path, env)
        # Don't update the normalization statistics during inference
        env.training = False
        # Don't normalize rewards - we want to see actual rewards
        env.norm_reward = False
    else:
        print(f"Warning: Normalization file {norm_path} not found!")

    # 5) Load model into the environment with new data
    model = DQN.load(model_path, env=env)

    # 6) Run one episode
    actions, rewards, info = run_episode(env, model)

    # 7) Process results
    df = df[["ID1_price", "Hour", "DayOfWeek", "Month"]]
    result_df = df.iloc[:len(actions)].reset_index(drop=True)
    actions_list = [int(a) if isinstance(a, np.ndarray) else a for a in actions]

    result_df["action"] = actions
    result_df["reward"] = rewards
    result_df["cumulative_reward"] = np.cumsum(rewards)

    # 8) Save to CSV
    output_path = f"{model_dir}/inference_results_in_sample.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Inference complete. Saved to {output_path}")
    print(f"Total reward: {sum(rewards):.2f}")

    # 9) Display summary statistics
    action_counts = pd.Series(actions_list).value_counts().to_dict()
    print("\n=== Summary Statistics ===")
    print(f"Action distribution: {action_counts}")
    print(f"Average reward per step: {np.mean(rewards):.4f}")
    print(f"Min reward: {np.min(rewards):.4f}, Max reward: {np.max(rewards):.4f}")
    print(f"Standard deviation of rewards: {np.std(rewards):.4f}")


if __name__ == "__main__":
    main()