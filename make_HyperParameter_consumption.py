import pandas as pd
import numpy as np


if __name__ == "__main__":
    loc_data = f"../data"

    df_frame = pd.DataFrame(index=range(1, 13, 1), columns=['mean', 'std'])

    for month in range(1, 13):
        sample_set = []
        sam1 = pd.read_excel(f"{loc_data}/시흥 스마트시티 데이터(비식별)_19_{month}.xlsx", sheet_name="누진2단계(201~300)").loc[:, "사용량 (kWH)"].values.tolist()
        sam2 = pd.read_excel(f"{loc_data}/시흥 스마트시티 데이터(비식별)_19_{month}.xlsx", sheet_name="누진2단계(301~400)").loc[:, "사용량 (kWH)"].values.tolist()
        sam3 = pd.read_excel(f"{loc_data}/시흥 스마트시티 데이터(비식별)_19_{month}.xlsx", sheet_name="누진3단계(401~)").loc[:, "사용량 (kWH)"].values.tolist()

        sample_set.append(sam1)
        sample_set.append(sam2)
        sample_set.append(sam3)

        sample_set = sum(sample_set, [])
        sample_set = np.array(sample_set)

        df_frame.loc[month, 'mean'] = np.mean(sample_set)
        df_frame.loc[month, 'std'] = np.std(sample_set)

    df_frame.to_excel(f"{loc_data}/electricity_consumption_normal_parameter.xlsx")