import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
import os

from battery_env import BatteryEnv


def run_episode(env: gym.Env, model: DQN):
    """
    Runs one episode in `env` using `model`, returns a list of actions and rewards.
    """
    obs, info = env.reset()  # Updated reset method for Gymnasium env
    done = False
    truncated = False

    actions, rewards = [], []
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)  # Updated step method for Gymnasium env

        actions.append(action)
        rewards.append(reward)

    return actions, rewards, info


def main():
    # 1) Load trained model
    model_path = "models/dqn_battery_34.zip"
    print(f"Loading model from {model_path}")

    # 2) Use a different dataset for inference
    # Using 2024 data instead of 2023 data
    data_path = "../data/preprocessed_data_2023.csv"
    print(f"Loading inference data from {data_path}")
    environment_df = pd.read_csv(data_path)

    # 3) Create environment with the new data
    env = BatteryEnv(environment_df)

    # 4) Load model into the environment with new data
    model = DQN.load(model_path, env=env)

    # 5) Run one episode
    actions, rewards, info = run_episode(env, model)

    # 6) Process results
    df = pd.read_csv(data_path)
    # Adjust to match your dataset columns
    df.columns = ["ID1_price", "Hour", "DayOfWeek", "Month"]

    # Ensure the results only cover the steps we've taken
    result_df = df.iloc[:len(actions)].reset_index(drop=True)

    # Convert numpy arrays to integers or appropriate scalar values
    actions_list = [int(a) if isinstance(a, np.ndarray) else a for a in actions]

    result_df["action"] = actions
    result_df["reward"] = rewards
    result_df["cumulative_reward"] = np.cumsum(rewards)

    # 7) Save to CSV
    output_path = "../data/inference_results_2023.csv"
    result_df.to_csv(output_path, index=False)
    print(f"✅ Inference complete. Saved to {output_path}")
    print(f"Total reward: {sum(rewards):.2f}")

    # 8) Display summary statistics - using the converted actions list
    action_counts = pd.Series(actions_list).value_counts().to_dict()
    print("\n=== Summary Statistics ===")
    print(f"Action distribution: {action_counts}")
    print(f"Average reward per step: {np.mean(rewards):.4f}")
    print(f"Min reward: {np.min(rewards):.4f}, Max reward: {np.max(rewards):.4f}")
    print(f"Standard deviation of rewards: {np.std(rewards):.4f}")


if __name__ == "__main__":
    main()