import pandas as pd


class Res:
    def __init__(self, user_info, auction_book, users_book, tariff_season):
        self.user_info = user_info
        self.auction_book = auction_book
        self.users_book = users_book
        self.tariff_season = tariff_season

        self.cal_user_info(user_info, users_book, auction_book)
        # self.cal_user_info_new(user_info, users_book, auction_book, tariff_season)
        # self.cal_utility()
    # 시간별 시장가격, 총 시장거래량, 총 낙찰량, 총 구매자 입찰가, 총 판매자 입찰가, 총 구매자 낙찰가, 총 판매자 낙찰가를 담는 df 필요

    def cal_user_info(self, user_info, users_book, auction_book):
        _index = user_info.keys()
        _column = [
            "순 조달량",
            "역송량",
            "판매량",
            "기본요금",
            "잔여 크레딧",
            "구매 낙찰량",
            "구매 낙찰가격",
            "판매 낙찰량",
            "판매 낙찰가격",
            "미 판매량",
            "SMP 보상요금",
        ]

        info_df = pd.DataFrame(index=_index, columns=_column)

        for u in _index:
            info_df.at[u, "순 조달량"] = sum(user_info[u]["Net_consumption"].values())
            info_df.at[u, "역송량"] = sum(user_info[u]['Feed_in'].values())
            info_df.at[u, "판매량"] = - sum([int(v) for v in user_info[u]["Feed_in"].values()])
            info_df.at[u, "기본요금"] = user_info[u]["Demand_charge"]

            amount = info_df.at[u, "순 조달량"] - info_df.at[u, "역송량"]
            if amount >= 0:
                info_df.at[u, "잔여 크레딧"] = 0

            else:
                info_df.at[u, "잔여 크레딧"] = -amount

            info_df.at[u, "구매 낙찰량"] = \
                sum([
                        sum(
                            users_book[date][u]["Q_winning"]
                        )
                        for date in users_book.keys()
                        if u in auction_book[date]["buyers"].keys()]
                )

            info_df.at[u, "구매 낙찰가격"] = \
                sum([
                        sum(
                            users_book[date][u]["P_winning"]
                        )
                        for date in users_book.keys()
                        if u in auction_book[date]["buyers"].keys()]
                )

            info_df.at[u, "판매 낙찰량"] = \
                sum([
                        sum(
                            users_book[date][u]["Q_winning"]
                        )
                        for date in users_book.keys()
                        if u in auction_book[date]["sellers"].keys()]
                )

            info_df.at[u, "판매 낙찰가격"] = \
                sum([
                        sum(
                            users_book[date][u]["P_winning"]
                        )
                        for date in users_book.keys()
                        if u in auction_book[date]["sellers"].keys()]
                )

            info_df.at[u, '미 판매량'] = sum(
                [
                    auction_book[date]["sellers"][u]["quantity"] -
                    sum(users_book[date][u]["Q_winning"])
                    for date in users_book.keys()
                    if u in auction_book[date]["sellers"].keys()
                ]
            )

            info_df.at[u, "SMP 보상요금"] = sum(
                [
                    auction_book[date]["market-price"] *
                    (
                            auction_book[date]["sellers"][u]["quantity"] -
                            sum(users_book[date][u]["Q_winning"])
                    )
                    for date in users_book.keys()
                    if u in auction_book[date]["sellers"].keys()
                ]
            )

        self.info_df = info_df

    def cal_user_info_new(self, user_info, users_book, auction_book, tariff_season):
        _index = user_info.keys()
        _column = [
            "구매 낙찰량",
            "구매 낙찰가격",
            "판매 낙찰량",
            "판매 낙찰가격",
            "미 판매량",
            "SMP 보상요금",
            "순 조달량",
            "누진단계",
            "기본요금",
            "전력량요금",
            "총요금"
        ]

        info_new_df = pd.DataFrame(index=_index, columns=_column)

        for u in _index:
            info_new_df.at[u, "구매 낙찰량"] = \
                sum([
                        sum(
                            users_book[date][u]["Q_winning"]
                        )
                        for date in users_book.keys()
                        if u in auction_book[date]["buyers"].keys()]
                )

            info_new_df.at[u, "구매 낙찰가격"] = \
                sum([
                        sum(
                            users_book[date][u]["P_winning"]
                        )
                        for date in users_book.keys()
                        if u in auction_book[date]["buyers"].keys()]
                )

            info_new_df.at[u, "판매 낙찰량"] = \
                sum([
                        sum(
                            users_book[date][u]["Q_winning"]
                        )
                        for date in users_book.keys()
                        if u in auction_book[date]["sellers"].keys()]
                )

            info_new_df.at[u, "판매 낙찰가격"] = \
                sum([
                        sum(
                            users_book[date][u]["P_winning"]
                        )
                        for date in users_book.keys()
                        if u in auction_book[date]["sellers"].keys()]
                )

            info_new_df.at[u, '미 판매량'] = sum(
                [
                    auction_book[date]["sellers"][u]["quantity"] -
                    sum(users_book[date][u]["Q_winning"])
                    for date in users_book.keys()
                    if u in auction_book[date]["sellers"].keys()
                ]
            )
            info_new_df.at[u, "SMP 보상요금"] = sum(
                [
                    auction_book[date]["market-price"] *
                    (
                            auction_book[date]["sellers"][u]["quantity"] -
                            sum(users_book[date][u]["Q_winning"])
                    )
                    for date in users_book.keys()
                    if u in auction_book[date]["sellers"].keys()
                ]
            )

            net_ToC = sum(user_info[u]["Net_consumption"].values()) - info_new_df.at[u, "구매 낙찰량"]
            info_new_df.at[u, "순 조달량"] = net_ToC
            info_new_df.at[u, "누진단계"] = tariff_season[net_ToC < tariff_season.loc[:, "max"]].iloc[0, :].name

            info_new_df.at[u, "기본요금"] = user_info[u]["Demand_charge"]

            block = tariff_season[net_ToC < tariff_season.loc[:, "max"]].iloc[0, :]
            block_lower = tariff_season[net_ToC >= tariff_season.loc[:, "max"]]
            energy_charge = 0
            max_val = 0
            for i in block_lower.index:
                energy_charge += (block_lower.loc[i, 'max'] - max_val) * block_lower.loc[i, f"{user_info[u]['Vol_level']}, 전력량"]
                max_val = block_lower.loc[i, 'max']
            energy_charge += (net_ToC - max_val) * block[f"{user_info[u]['Vol_level']}, 전력량"]

            info_new_df.at[u, "전력량요금"] = energy_charge

            info_new_df.at[u, "총요금"] = \
                info_new_df.at[u, "구매 낙찰가격"] + \
                info_new_df.at[u, "판매 낙찰가격"] + \
                info_new_df.at[u, "전력량요금"] + \
                info_new_df.at[u, "기본요금"] + \
                info_new_df.at[u, "SMP 보상요금"]

        self.user_info_new = info_new_df

    def cal_utility(self):
        _index = [
            '0'
        ]
        _column = [
            "상계거래 후 전기요금",
            "auction 후 전기요금",
            "auction 손실금"
        ]
        info_df = pd.DataFrame(index=_index, columns=_column)

        for u in _index:
            info_df.at[u, "상계거래 후 전기요금"] = self.user_info_old['총요금'].sum()

            info_df.at[u, "auction 후 전기요금"] = self.user_info_new["전력량요금"].sum() + self.user_info_new["기본요금"].sum()

            info_df.at[u, "auction 손실금"] = - (self.user_info_new["구매 낙찰가격"].sum() +
                                              self.user_info_new["판매 낙찰가격"].sum())
        self.utility_info = info_df



