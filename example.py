import pandas as pd
import numpy as np
from inf_fs import inf_fs, select_inf_fs

df = pd.read_csv('breast-cancer-wisconsin.data', header=None).iloc[:,1:].replace('?', np.nan).dropna().astype(int)

rank, energy = inf_fs(df.to_numpy())

print('rank', rank)
print('energy', energy)

reduced = select_inf_fs(df.to_numpy(), 3)
print(reduced[:10])