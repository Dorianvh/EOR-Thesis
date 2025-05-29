import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BatteryEnv(gym.Env):
    """Custom Environment for Battery Storage Arbitrage using gymnasium."""

    def __init__(self, environment_df = None):
        super(BatteryEnv, self).__init__()

        # Save price column for reward calculation
        self.prices = environment_df['ID1_price']

        # select only the columns in the observation space
        self.observations_df = environment_df[['ID1_price', 'Hour', 'DayOfWeek', 'Month']]

        # Each step for the agent corresponds to one row of data
        self.max_step = len(self.observations_df) - 1

        # Battery parameters
        self.battery_capacity = 1.0  # in MWh
        self.max_power = 1.0        # max charge/discharge per step (MW)
        self.round_way_efficiency = 0.9
        self.one_way_efficiency = np.sqrt(self.round_way_efficiency)  # charge/discharge efficiency
        self.soc_min = 0.2  # minimum state of charge
        self.soc_max = 0.8  # maximum state of charge
        self.invalid_action_penalty = -10.0  # penalty for invalid actions

        # Define observation space: all columns of the DataFrame
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observations_df.shape[1] + 1,),  # Add 1 for the SoC
            dtype=np.float32
        )

        # Define action space: 0 = charge, 1 = hold, 2 = discharge
        self.action_space = spaces.Discrete(3)

        #print("[INIT] Environment initialized with", len(self.df), "data steps.")

    def reset(self, *, seed=None, options=None):
        """Resets the environment to the starting state."""
        #print("[RESET] Resetting environment...")

        self.step_idx = 0
        self.soc = self.soc_min  # start with the minimum SoC
        observation = self._get_obs()
        info = {}

        print(f"[RESET] Starting at step {self.step_idx}, SoC: {self.soc:.2f}")
        return observation, info

    def _get_obs(self):
        """Returns current observation: all columns of the current row."""
        current_row = self.observations_df.iloc[self.step_idx].to_numpy(dtype=np.float32)
        return np.append(current_row, self.soc)

    def step(self, action):
        """Takes one step in the environment based on the selected action."""
        price = self.prices.iloc[self.step_idx]
        reward = 0.0

        #print(f"[STEP] Step {self.step_idx} | Price: {price:.2f} | SoC: {self.soc:.2f} | Action: {action}")

        if action == 0:  # CHARGE
            if self.soc >= self.soc_max:
                reward = self.invalid_action_penalty
                #print(f"[INVALID ACTION] Cannot charge: SoC is already at maximum, SoC: {self.soc:.2f}, reward: {reward:.2f}")
            else:
                energy = -((self.soc_max - self.soc) / self.one_way_efficiency) * self.battery_capacity
                cost = energy * price
                self.soc = self.soc - ((-self.one_way_efficiency * np.abs(energy)) / self.battery_capacity)
                reward = cost
                #print(f"[ACTION] Charging: bought {energy:.3f} MWh, new SoC: {self.soc:.2f}, reward: {reward:.2f}")

        elif action == 2:  # DISCHARGE
            if self.soc <= self.soc_min:
                reward = self.invalid_action_penalty
                #print(f"[INVALID ACTION] Cannot discharge: SoC is already at minimum, SoC: {self.soc:.2f}, reward: {reward:.2f}")
            else:
                energy = self.one_way_efficiency * (self.soc - self.soc_min) * self.battery_capacity
                revenue = energy * price
                self.soc = self.soc - ((np.abs(energy) / self.one_way_efficiency) / self.battery_capacity)
                reward = revenue
                #print(f"[ACTION] Discharging: sold {energy:.3f} MWh, new SoC: {self.soc:.2f}, reward: {reward:.2f}")

        else:
            reward = 0.0
            #print("[ACTION] Hold: no battery action taken.")

        self.step_idx += 1
        done = self.step_idx >= self.max_step
        truncated = False  # No truncation logic in this environment
        obs = self._get_obs()
        info = {}

        return obs, reward, done, truncated, info



