import os
import pandas as pd
import json


if __name__ == "__main__":
    loc_data = f"../data"

    pv_hist_list = os.listdir(f"{loc_data}/pv")

    sample_dict = dict()
    for i in range(365):
        for j in range(24):
            sample_dict[f"{i},{j}"] = []
    for f in pv_hist_list:
        pv_f = pd.read_excel(f"{loc_data}/pv/{f}", sheet_name='Sheet1', index_col=0)
        pv_cap_f = pd.read_excel(f"{loc_data}/pv/{f}", sheet_name='Sheet2').iloc[0, 0]
        pv_f = pv_f.values / pv_cap_f
        for i in range(365):
            for j in range(24):
                sample_dict[f"{i},{j}"].append(pv_f[i, j])

    with open(f"{loc_data}/pv_gen_sample_set.json", 'w') as f:
        json.dump(sample_dict, f)