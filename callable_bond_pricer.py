"""
Interest Rate Models - Black Derman Toy

Source Code

Author : William Carpenter
Date   : April 2024

Objective: Price callable bonds with BDT model.

"""
import sys
import os
import math 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
# Custom module
import model_bdt as model

#%%
zero_cpns = pd.read_csv('https://raw.githubusercontent.com/wrcarpenter/Interest-Rate-Models/main/Data/zcbs.csv') 
zcbs      = zero_cpns.loc[zero_coupons['Date']=='3/8/2024']
zcbs      = zcbs.drop("Date", axis=1)

# small example for calibration
zeros    = np.array(zcbs.iloc[:,0:24])
x        = model.build(zeros, 0.14, 1/12) 
tree_bdt = model.rateTree(x[0], x[2], 0.14, 1/12)

#%% 
bcf      = cf_bond(tree_bdt, 5.00, 1/12, 100, 5.00)
pricing  = priceTree(tree_bdt, 1/2, bcf, 1/12, "bond", 100)
oprice   = priceOption(tree_bdt, 1/2, bcf, 1/12, "bond", 100, pricing[1], pricing[0])

print(pricing[0])
ptree = pricing[1]
otree = oprice[3]

#%%
print(zeros[0,zeros.shape[1]-1])
print(pricing[0])
print(pricing[0] - zeros[0,zeros.shape[1]-1])












