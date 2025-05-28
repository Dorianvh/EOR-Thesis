import numpy as np

data = np.load("../logs/evaluations_34.npz", allow_pickle=True)
lst = data.files

for item in lst:
    print(item)
    print(data[item])