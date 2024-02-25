'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2024-01-26 14:08:29
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2024-01-26 14:33:38
FilePath: /DatProc/run_single_image_pipeline.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
#%%
import os
from dpmain.datproc_v1 import DatProcV1
dp = DatProcV1(data_source='rebuttal')
# %%
image_root = '/data2/SphericalHead/Rebuttal/front'
for f_name in os.listdir(image_root):
    f_path = os.path.join(image_root, f_name)
    results = dp(f_path)
    print(f_name, end=':')
    print(results[0]['head']['camera'])