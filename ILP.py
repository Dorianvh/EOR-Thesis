import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

df = pd.read_csv('data/preprocessed_data_2024.csv')
prices = df['ID1_price']
T = len(prices)


soc_low = 0.2
soc_high = 0.8
initial_soc = 0.2
eta = 0.9          # round-way efficiency
delta_soc = soc_high - soc_low


m = gp.Model("BatteryArbitrageILP")


is_high = m.addVars(T + 1, vtype=GRB.BINARY, name="isHigh")       # SoC state
charge = m.addVars(T, vtype=GRB.BINARY, name="Charge")
discharge = m.addVars(T, vtype=GRB.BINARY, name="Discharge")
idle = m.addVars(T, vtype=GRB.BINARY, name="Idle")


if initial_soc == soc_high:
    m.addConstr(is_high[0] == 1, name="InitialSOC")
else:
    m.addConstr(is_high[0] == 0, name="InitialSOC")


for t in range(T):
    # Only one action per step
    m.addConstr(charge[t] + discharge[t] + idle[t] == 1, name=f"OneAction_{t}")

    m.addConstr(
        is_high[t + 1] ==
        is_high[t] * idle[t] +  # stays the same if idle
        1 * charge[t] +  # becomes high if charging
        0 * discharge[t],  # becomes low if discharging (do nothing)
        name=f"SoCTransition_{t}"
    )

    # Only charge when SoC is 0.2
    m.addConstr(charge[t] <= 1 - is_high[t], name=f"OnlyChargeFromLow_{t}")

    # Only discharge when SoC is 0.8
    m.addConstr(discharge[t] <= is_high[t], name=f"OnlyDischargeFromHigh_{t}")

# Objective maximize profit
profit = gp.quicksum(
    prices[t] * (discharge[t] * delta_soc * np.sqrt(eta) - charge[t] * delta_soc / np.sqrt(eta))
    for t in range(T)
)

m.setObjective(profit, GRB.MAXIMIZE)


m.optimize()


if m.status == GRB.OPTIMAL:
    print(f"\n Optimal profit: €{m.objVal:.2f}\n")
    print("t   Action       Price     SoC")
    print("===============================")
    for t in range(min(50, T)):
        soc_level = soc_high if is_high[t].X > 0.5 else soc_low
        if charge[t].X > 0.5:
            action = "CHARGE"
        elif discharge[t].X > 0.5:
            action = "DISCHARGE"
        else:
            action = "IDLE"
        print(f"{t:2d}  {action:<10} €{prices[t]:>6.2f}   SoC = {soc_level:.1f}")
else:
    print(" No feasible solution found.")
