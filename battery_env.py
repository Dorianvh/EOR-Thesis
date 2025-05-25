import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class BatteryEnv(gym.Env):
    """Custom Environment for Battery Storage Arbitrage using gymnasium."""

    def __init__(self, price_file="data/ID1_prices_germany.csv"):
        super(BatteryEnv, self).__init__()

        print("[INIT] Loading electricity price data from:", price_file)

        # Load hourly electricity prices from CSV
        df = pd.read_csv(price_file, parse_dates=["Date"])
        df = df.dropna(subset=["ID1_price"])
        self.prices = df['ID1_price'].values
        self.max_step = len(self.prices) - 1

        # Battery parameters
        self.battery_capacity = 1.0  # in MWh
        self.max_power = 0.25        # max charge/discharge per step (MW)
        self.efficiency = 0.92       # charge/discharge efficiency

        # Define observation space: [current_price, state_of_charge]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

        # Define action space: 0 = charge, 1 = hold, 2 = discharge
        self.action_space = spaces.Discrete(3)

        print("[INIT] Environment initialized with", len(self.prices), "price steps.")

    def reset(self, *, seed=None, options=None):
        """Resets the environment to the starting state."""
        print("[RESET] Resetting environment...")

        self.step_idx = 0
        self.soc = 0.5  # start with 50% state of charge
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
            energy = min(self.max_power, self.battery_capacity - self.soc)
            cost = energy * price
            self.soc += energy * self.efficiency
            reward = -cost
            print(f"[ACTION] Charging: bought {energy:.3f} MWh, new SoC: {self.soc:.2f}, reward: {reward:.2f}")

        elif action == 2:  # DISCHARGE
            energy = min(self.max_power, self.soc)
            revenue = energy * price
            self.soc -= energy / self.efficiency
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

    def render(self):
        """Optional: Render current state (not required here)."""
        print(f"[RENDER] Step {self.step_idx}, Price: {self.prices[self.step_idx]}, SoC: {self.soc}")
