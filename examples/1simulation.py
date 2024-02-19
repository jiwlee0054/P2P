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
    input_data_name = "casestudy_20_pv10_10_low"

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





    """
    note

    1. 낙찰량이 많은 판매자는 편익 증가율이 높다
    2. 낙찰량이 낮은 판매자는 구매자보다 편익 증가율이 낮다.
    3. 낙찰률이 높게 되도록 유인된다. 판매자의 낙찰률이 높으려면, 구매자의 입찰량이 많아야 함
    4. 누진단계가 낮을수록 낙찰률이 높고, 편익이 상당히 증가한다. >> 중요 !
    5. 구매자는 누진단계가 낮을수록 낙찰률이 낮고, 편익이 0에 가깝다.
    6. 
    """