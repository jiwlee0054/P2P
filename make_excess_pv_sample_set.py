import pandas as pd
import json


if __name__ == '__main__':
    loc_data = f"../data"

    excess_electricity_sample_set = pd.read_excel(f"{loc_data}/excess_electricity.xlsx", sheet_name='Sheet1')

    set_dict = dict()
    for i in excess_electricity_sample_set.columns[1: ]:
        set_dict[i] = excess_electricity_sample_set.loc[:, i].values.tolist()

    with open(f"{loc_data}/excess_electricity_sample_set.json", 'w') as f:
        json.dump(set_dict, f)