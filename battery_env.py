import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class BatteryEnv(gym.Env):
    """Custom Environment for Battery Storage Arbitrage using gymnasium."""

    def __init__(self, data_file="data/ID1_prices_germany.csv"):
        super(BatteryEnv, self).__init__()

        print("[INIT] Loading environment data from:", data_file)

        # Load hourly data from csv, this will be the external environment
        df = pd.read_csv(data_file, parse_dates=["Date"])
        df = df.dropna(subset=["ID1_price"])
        self.prices = df['ID1_price'].values

        # Each step for the agent corresponds to one hour of data
        self.max_step = len(self.prices) - 1

        # Battery parameters
        self.battery_capacity = 1.0  # in MWh
        self.max_power = 1.0        # max charge/discharge per step (MW)
        self.round_way_efficiency = 0.9
        self.one_way_efficiency = np.sqrt(self.round_way_efficiency)  # charge/discharge efficiency
        self.soc_min = 0.2  # minimum state of charge
        self.soc_max = 0.8  # maximum state of charge
        self.invalid_action_penalty = -100.0  # penalty for invalid actions

        # Define observation space: [current_price, state_of_charge]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

        # Define action space: 0 = charge, 1 = hold, 2 = discharge
        self.action_space = spaces.Discrete(3)

        print("[INIT] Environment initialized with", len(self.prices), "price steps.")

    def reset(self, *, seed=None, options=None):
        """Resets the environment to the starting state."""
        print("[RESET] Resetting environment...")

        self.step_idx = 0
        self.soc = self.soc_min  # start with the minimun soc
        observation = self._get_obs()
        info = {}

        print(f"[RESET] Starting at price: {self.prices[self.step_idx]:.2f}, SoC: {self.soc:.2f}")
        return observation, info

    def _get_obs(self):
        """Returns current observation: [price, state of charge]."""
        return np.array([self.prices[self.step_idx], self.soc], dtype=np.float32)

    def step(self, action):
        """Takes one step in the environment based on the selected action."""
        price = self.prices[self.step_idx]
        reward = 0.0

        print(f"[STEP] Step {self.step_idx} | Price: {price:.2f} | SoC: {self.soc:.2f} | Action: {action}")

        if action == 0:  # CHARGE
            if self.soc >= self.soc_max:
                reward = self.invalid_action_penalty
                print(f"[INVALID ACTION] Cannot charge: SoC is already at maximum, SoC: {self.soc:.2f}, reward: {reward:.2f}")
            else:
                energy = -((self.soc_max - self.soc) / self.one_way_efficiency) * self.battery_capacity
                cost = energy * price
                self.soc = self.soc - ((-self.one_way_efficiency * np.abs(energy)) / self.battery_capacity)
                reward = cost
                print(f"[ACTION] Charging: bought {energy:.3f} MWh, new SoC: {self.soc:.2f}, reward: {reward:.2f}")


        elif action == 2:  # DISCHARGE
            if self.soc <= self.soc_min:
                reward = self.invalid_action_penalty
                print(f"[INVALID ACTION] Cannot discharge: SoC is already at minimum, SoC: {self.soc:.2f}, reward: {reward:.2f}")
            else:
                energy = self.one_way_efficiency * (self.soc - self.soc_min) * self.battery_capacity
                revenue = energy * price
                self.soc = self.soc - ((np.abs(energy) / self.one_way_efficiency) / self.battery_capacity)
                reward = revenue
                print(f"[ACTION] Discharging: sold {energy:.3f} MWh, new SoC: {self.soc:.2f}, reward: {reward:.2f}")


        else:
            print("[ACTION] Hold: no battery action taken.")

        self.step_idx += 1
        done = self.step_idx >= self.max_step
        truncated = False  # No truncation logic in this environment
        obs = self._get_obs()
        info = {}

        return obs, reward, done, truncated, info



