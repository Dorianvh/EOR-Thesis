import pandas as pd
import multiprocessing
import os
import glob
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from battery_env import BatteryEnv


def create_run_directory():
    """Create a uniquely numbered directory for this training run."""
    # Find the next run number by checking existing directories
    runs_base_dir = "./training_runs"
    os.makedirs(runs_base_dir, exist_ok=True)

    existing_runs = glob.glob(f"{runs_base_dir}/run_*")
    run_numbers = [int(run.split("_")[-1]) for run in existing_runs if run.split("_")[-1].isdigit()]
    next_run_number = 1 if not run_numbers else max(run_numbers) + 1

    # Create the run directory
    run_dir = f"{runs_base_dir}/run_{next_run_number}"
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def save_hyperparameters(run_dir, hyperparams, total_timesteps, feature_columns):
    """Save hyperparameters to a text file."""
    # Add total timesteps to the parameters
    params_dict = {**hyperparams, "total_timesteps": total_timesteps
                   , "feature_columns": feature_columns}

    # Save as a formatted text file
    with open(f"{run_dir}/hyperparameters.txt", "w") as f:
        for key, value in params_dict.items():
            f.write(f"{key}: {value}\n")


def train(env_df, feature_columns):
    """Train the DQN agent on the battery environment."""
    # Create a unique directory for this run
    run_dir = create_run_directory()
    print(f"[TRAINING] Saving all results to: {run_dir}")

    # Create a vectorized environment with multiple parallel environments
    num_envs = 14  # Adjust based on your CPU cores
    env = make_vec_env(
        lambda: BatteryEnv(env_df,feature_columns),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv  # Use SubprocVecEnv for true parallelism
    )

    # For evaluation, we can use a single environment
    eval_env = make_vec_env(lambda: BatteryEnv(env_df, feature_columns), n_envs=1)

    print("[TRAINING] Creating DQN model...")
    # Store hyperparameters for later saving
    hyperparams = {
        "policy": "MlpPolicy",
        "learning_rate": 1e-3,
        "buffer_size": int(100000 * (num_envs ** 0.5)),
        "learning_starts": 1000 * num_envs,
        "batch_size": 64,
        "gamma": 0.99,
        "target_update_interval": int(250 * (num_envs ** 0.5)),
        "gradient_steps": num_envs,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "num_envs": num_envs
    }

    model = DQN(
        policy=hyperparams["policy"],
        env=env,
        verbose=1,
        learning_rate=hyperparams["learning_rate"],
        buffer_size=hyperparams["buffer_size"],
        learning_starts=hyperparams["learning_starts"],
        batch_size=hyperparams["batch_size"],
        gamma=hyperparams["gamma"],
        target_update_interval=hyperparams["target_update_interval"],
        gradient_steps=hyperparams["gradient_steps"],
        tensorboard_log=f"{run_dir}/",
        exploration_fraction=hyperparams["exploration_fraction"],
        exploration_initial_eps=hyperparams["exploration_initial_eps"],
        exploration_final_eps=hyperparams["exploration_final_eps"]
    )

    print("[TRAINING] Defining evaluation callback...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,  # Save directly in run directory
        log_path=run_dir,  # Save logs directly in run directory
        eval_freq=10000 // num_envs,  # Adjust for vectorized environment
        n_eval_episodes=1,
        deterministic=True,
        render=False
    )

    # Total timesteps for training
    total_timesteps = 5000000

    # Save hyperparameters before training
    save_hyperparameters(run_dir, hyperparams, total_timesteps, feature_columns)

    print("[TRAINING] Starting training")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save the final model
    final_model_path = f"{run_dir}/final_model"
    model.save(final_model_path)
    print(f"[TRAINING] Final model saved to {final_model_path}")

    print(f"[TRAINING] Training complete. All results saved to {run_dir}")


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Needed for Windows

    # Load the environment data from csv
    env_df = pd.read_csv("../data/preprocessed_data_2023.csv")

    # Columns in env df that contain the features for the agent
    #feature_columns = ['ID1_price']
    feature_columns = ['ID1_price', 'Hour', 'DayOfWeek', 'Month']
    #feature_columns = ['ID1_price_rolling_z_score', 'Hour', 'DayOfWeek', 'Month']

    train(env_df, feature_columns)