import pandas as pd
import multiprocessing
import os
import glob
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from battery_env import BatteryEnv
import json


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
            if isinstance(value, dict):
                f.write(f"{key}: {json.dumps(value, indent=2)}\n")
            else:
                f.write(f"{key}: {value}\n")


def train(env_df, feature_columns):
    """Train the DQN agent on the battery environment."""
    run_dir = create_run_directory()
    print(f"[TRAINING] Saving all results to: {run_dir}")

    num_envs = 10

    # Base vectorized envs
    base_env = make_vec_env(
        lambda: BatteryEnv(env_df, feature_columns),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv
    )

    base_eval_env = make_vec_env(
        lambda: BatteryEnv(env_df, feature_columns),
        n_envs=1,
        vec_env_cls=SubprocVecEnv
    )

    # Wrap with VecNormalize for observation normalization
    env = VecNormalize(base_env, norm_obs=True, norm_reward=False)
    eval_env = VecNormalize(base_eval_env, norm_obs=True, norm_reward=True)

    # Same as before
    net_arch = {
        "type": "MLP",
        "layers": [64,64],
        "activation": "ReLU"
    }

    hyperparams = {
        "policy": "MlpPolicy",
        "network_architecture": net_arch,
        "learning_rate": 1e-4,
        "buffer_size": int(10000 * (num_envs ** 0.5)),
        "learning_starts": 1000 * num_envs,
        "batch_size": 32,
        "gamma": 0.99,
        "target_update_interval": int(250 * (num_envs ** 0.5)),
        "gradient_steps": num_envs,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "num_envs": num_envs
    }

    policy_kwargs = dict(net_arch=net_arch["layers"])

    model = DQN(
        policy=hyperparams["policy"],
        env=env,
        verbose=1,
        policy_kwargs=policy_kwargs,
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
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=10000 // num_envs,
        n_eval_episodes=1,
        deterministic=True,
        render=False
    )

    total_timesteps = 1000000
    save_hyperparameters(run_dir, hyperparams, total_timesteps, feature_columns)

    print("[TRAINING] Starting training")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save model and VecNormalize statistics
    final_model_path = f"{run_dir}/final_model"
    model.save(final_model_path)
    env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    print(f"[TRAINING] Final model saved to {final_model_path}")
    print(f"[TRAINING] VecNormalize statistics saved to {run_dir}/vecnormalize.pkl")
    print(f"[TRAINING] Training complete. All results saved to {run_dir}")


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Needed for Windows

    # Load the environment data from csv
    env_df = pd.read_csv("../data/scaled_data_2023.csv")

    # Columns in env df that contain the features for the agent
    #feature_columns = ['ID1_price']
    feature_columns = ['ID1_price', 'Hour', 'DayOfWeek', 'Month',
                       'ARMAX_forecast_1hour', 'ARMAX_forecast_3hour']
    #feature_columns = ['ID1_price_scaled','ARMAX_forecast_1hour_scaled', 'ARMAX_forecast_3hour_scaled']

    train(env_df, feature_columns)