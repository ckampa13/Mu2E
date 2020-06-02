import os
import time
import pandas as pd
import pickle as pkl

from mu2e import mu2e_ext_path
from hallprobesim_redux import *
from mu2e.fieldfitter_redux2 import FieldFitter

df_Mu2e = pd.read_pickle(mu2e_ext_path+'Bmaps/Mu2e_DSMap_V13.p')
df_Mu2e = df_Mu2e.query("(R >= 25e-3) & (R <= 0.8) & (Z >= 4.2) & (Z <= 13.9)")

cfg_pickle_Mau13_recreate = cfg_pickle(use_pickle=True, save_pickle=False, load_name='Mau13',save_name='Mau13', recreate=True)

ff = FieldFitter(df_Mu2e.sample(100), cfg_geom_cyl_800mm_long)
ff.merge_data_fit_res()

ff.input_data.to_pickle(mu2e_ext_path+'Bmaps/Mau13_validation_df.p')
