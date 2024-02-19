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

    for n in num_list:
        options = ProbOptions(input_data_name)
        IFN = ReadInputData(options)
        ST = SetTimes()
        UI = UserInfo(options)

        for m in range(1, 13, 1):
            options.month = m
            ST.main(options, IFN)
            UI.main(options, IFN, ST)

            auction_book, users_book, historic_bid = market_clearing(
                options,
                IFN,
                ST,
                UI
            )

            UI.auction_book.update(auction_book)
            UI.users_book.update(users_book)
            UI.historic_bid.update(historic_bid)

            UI.res_info(options, ST)

        concat_list = []
        for m in UI.res_book.keys():
            if 'month' in UI.res_book[m].columns:
                UI.res_book[m].drop(['month'], axis=1, inplace=True)
            UI.res_book[m].insert(0, 'month', m)
            concat_list.append(UI.res_book[m])
        df_all = pd.concat(concat_list, axis=0)

        df_benefit.loc["프로슈머 편익(종합)", n] = df_all[df_all["설치용량"] > 0]["편익"].sum()
        df_benefit.loc["컨슈머 편익(종합)", n] = df_all[df_all["설치용량"] == 0]["편익"].sum()
        df_benefit.loc["프로슈머-컨슈머 편익차이(종합)", n] = \
            df_benefit.loc["프로슈머 편익(종합)", n] - df_benefit.loc["컨슈머 편익(종합)", n]
        df_benefit.loc["유틸리티 손실(옥션)", n] = df_all["구매 낙찰가격"].sum() + df_all["판매 낙찰가격"].sum()
        df_benefit.loc["유틸리티 수익(옥션)", n] = \
            df_all["옥션 후 SMP 보상요금"].sum() + \
            df_all["기본요금"].sum() + \
            df_all["옥션 후 전력량요금"].sum() + \
            df_benefit.loc["유틸리티 손실(옥션)", n]
        df_benefit.loc["유틸리티 수익(상계거래)", n] = \
            df_all["상계거래 후 잔여크레딧 보상요금"].sum() + \
            df_all["기본요금"].sum() + \
            df_all["상계거래 후 전력량요금"].sum()
        df_benefit.loc["유틸리티 편익(종합)", n] = \
            df_benefit.loc["유틸리티 수익(옥션)", n] - df_benefit.loc["유틸리티 수익(상계거래)", n]
        df_benefit.loc["유틸리티 수익 변화율", n] = \
            (df_benefit.loc["유틸리티 수익(옥션)", n] - df_benefit.loc["유틸리티 수익(상계거래)", n]) / \
            df_benefit.loc["유틸리티 수익(상계거래)", n] * 100
        df_benefit.loc["역송량", n] = df_all["역송량"].sum()
        df_benefit.loc["판매량", n] = df_all["판매량"].sum()
        df_benefit.loc["판매 낙찰량", n] = df_all["판매 낙찰량"].sum()

        df = pd.DataFrame(index=UI.users_book.keys(), columns=["SMP", "평균 판매자 낙찰가격", "최대 판매자 낙찰가격", "최소 판매자 낙찰가격"])
        for date in df.index:
            df.loc[date, "SMP"] = UI.auction_book[date]["market-price"]
            _p = sum([UI.users_book[date][u]["P_winning"] for u in UI.users_book[date]], [])
            _p_seller = [abs(k) for k in _p if k < 0]
            if len(_p_seller) > 0:
                df.loc[date, "평균 판매자 낙찰가격"] = sum(_p_seller) / len(_p_seller)
                df.loc[date, "최소 판매자 낙찰가격"] = min(_p_seller)
                df.loc[date, "최대 판매자 낙찰가격"] = max(_p_seller)
            else:
                pass
        df_all.to_excel(f"results/number-{len(UI.Users.keys())}_month-all_results_{options.year}_{input_data_name}_{n}.xlsx")
        # df.dropna(axis=0).to_excel(f"results/number-{len(UI.Users.keys())}_month-all_{options.year}_{input_data_name}_Case1_SMP_WP_{n}.xlsx")
        if n == 1:
            df_price["SMP"] = df["SMP"]
        df_price[n] = df.loc[df.index, "평균 판매자 낙찰가격"]

    df_benefit.to_excel(f"results/number-{len(UI.Users.keys())}_month-all_results_{options.year}_{input_data_name}_benefit 정리.xlsx")
    df_price.to_excel(f"results/number-{len(UI.Users.keys())}_month-all_results_{options.year}_{input_data_name}_판매자 평균 낙찰가격 정리.xlsx")


    """
    note
    
    1. 낙찰량이 많은 판매자는 편익 증가율이 높다
    2. 낙찰량이 낮은 판매자는 구매자보다 편익 증가율이 낮다.
    3. 낙찰률이 높게 되도록 유인된다. 판매자의 낙찰률이 높으려면, 구매자의 입찰량이 많아야 함
    4. 누진단계가 낮을수록 낙찰률이 높고, 편익이 상당히 증가한다. >> 중요 !
    5. 구매자는 누진단계가 낮을수록 낙찰률이 낮고, 편익이 0에 가깝다.
    6. 
    """