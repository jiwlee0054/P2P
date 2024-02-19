import pandas as pd

from lib.parameters import (
    ProbOptions,
    ReadInputData,
    SetTimes,
    UserInfo
)

from lib.auction_oper_model import (
    market_clearing
)

if __name__ == '__main__':
    input_data_name = "casestudy_20_pv10_10_Low"

    num_list = range(1, 11, 1)
    df_benefit = pd.DataFrame(index=["프로슈머 편익(종합)", "컨슈머 편익(종합)", "프로슈머-컨슈머 편익차이(종합)",
                                     "유틸리티 손실(옥션)", "유틸리티 수익(옥션)", "유틸리티 수익(상계거래)",
                                     "유틸리티 편익(종합)", "유틸리티 수익 변화율", "역송량", "판매량", "판매 낙찰량"], columns=num_list)
    df_price = pd.DataFrame(columns=num_list)

    options = ProbOptions(input_data_name)
    IFN = ReadInputData(options)
    ST = SetTimes()
    UI = UserInfo(options)

    profile_load = pd.DataFrame(columns=UI.Users.keys())
    profile_gen = pd.DataFrame(columns=UI.Users.keys())
    for m in range(1, 13, 1):
        options.month = m
        ST.main(options, IFN)
        UI.main(options, IFN, ST)

        for T in ST.date_list_resol:
            load_by_user = pd.DataFrame([sum(UI.Users[u]["Electricity_Consumption"][(T[0], t)] for t in ST.trading_time[T[1]]) for u in UI.Users.keys()], index=UI.Users.keys(), columns=[f"{m}, {T[0]}, {T[1]}"])
            load_by_user = load_by_user.transpose()
            profile_load = pd.concat((profile_load, load_by_user), axis=0)

            gen_by_user = pd.DataFrame([sum(UI.Users[u]["Generation"][(T[0], t)] for t in ST.trading_time[T[1]]) for u in UI.Users.keys()], index=UI.Users.keys(), columns=[f"{m}, {T[0]}, {T[1]}"])
            gen_by_user = gen_by_user.transpose()
            profile_gen = pd.concat((profile_gen, gen_by_user), axis=0)
    profile_load.to_excel(f"results/load profile.xlsx")
    profile_gen.to_excel(f"results/gen profile.xlsx")
