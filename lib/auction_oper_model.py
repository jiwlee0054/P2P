import copy
import random
from datetime import timedelta, datetime

import numpy as np
import pandas as pd

import lib.parameters as para
from lib.parameters import (
    ProbOptions,
    ReadInputData,
    SetTimes,
    UserInfo
)


def _set_block_step(net_ToC, tariff_season):
    tariff_block = tariff_season[net_ToC < tariff_season.loc[:, "max"]].iloc[0, :]
    return tariff_block.name


def _bid_by_strategy(
        options:ProbOptions,
        player_info,
        strategy,
        clearing_price=None
):
    def _bid_randomly(player_info):
        bid_pool = dict()
        for u in random.sample(list(player_info.keys()), len(player_info)):
            for i in range(abs(player_info[u]['quantity'])):
                bid_pool[f"{u},{i},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"] = \
                    np.random.uniform(player_info[u]['lower_price'], player_info[u]['upper_price'])
        return pd.DataFrame(bid_pool.values(), index=bid_pool.keys(), columns=['bid'])

    def _bid_update(player_info):
        """
        if historic bid >= historic clearing price
         - buyer:
        """
        bid_pool = dict()
        for u in random.sample(list(player_info.keys()), len(player_info)):
            if player_info[u]['historic_bid'] is np.nan:
                pass

            elif clearing_price <= player_info[u]['historic_bid'] <= player_info[u]['upper_price']:
                adjusted = \
                    player_info[u]['historic_bid'] - \
                    (player_info[u]['historic_bid'] - clearing_price) * options.extract_margin_rate()
                for i in range(abs(player_info[u]['quantity'])):
                    bid_pool[f"{u},{i},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"] = \
                        np.random.uniform(
                            low=player_info[u]['lower_price'],
                            high=adjusted
                        )

            elif player_info[u]['historic_bid'] <= clearing_price <= player_info[u]['upper_price']:
                adjusted = \
                    player_info[u]['historic_bid'] + \
                    (clearing_price - player_info[u]['historic_bid']) * options.extract_margin_rate()
                for i in range(abs(player_info[u]['quantity'])):
                    bid_pool[f"{u},{i},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"] = \
                        np.random.uniform(
                            low=adjusted,
                            high=player_info[u]['upper_price']
                        )

            elif player_info[u]['historic_bid'] <= player_info[u]['upper_price'] <= clearing_price:
                adjusted = \
                    player_info[u]['historic_bid'] + \
                    (clearing_price - player_info[u]['historic_bid']) * options.extract_margin_rate()
                # 조절된 값이 상한 입찰가를 초과할 경우, 입찰가는 상한입찰가로 입찰하도록 설정
                if adjusted > player_info[u]['upper_price']:
                    adjusted = player_info[u]['upper_price']
                for i in range(abs(player_info[u]['quantity'])):
                    bid_pool[f"{u},{i},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"] = \
                        np.random.uniform(
                            low=adjusted,
                            high=player_info[u]['upper_price']
                        )

            else:
                print("error: 알 수 없는 조건")
                print(f"user:{u}에서 문제 발생")

        if bid_pool == dict():
            return pd.DataFrame([], index=bid_pool.keys(), columns=['bid'])
        else:
            return pd.DataFrame(bid_pool.values(), index=bid_pool.keys(), columns=['bid'])

    if strategy == 'Randomly':
        return _bid_randomly(player_info)

    elif strategy == 'Update margin':
        if clearing_price is None:
            return _bid_randomly(player_info)
        else:
            return _bid_update(player_info)


def _market_price(
        auction_mechanism,
        bid_buyer_pool,
        bid_seller_pool
):

    # breakeven index 결정 (동일 index의 구매자의 입찰가가 판매자의 입찰가보다 큰 것 중 가장 높은 index)
    if bid_buyer_pool.shape[0] >= bid_seller_pool.shape[0]:
        pass_margin = bid_seller_pool.shape[0]
        breakeven_index = np.where(bid_buyer_pool.iloc[:pass_margin, 0].values >= bid_seller_pool.iloc[:, 0].values)[0][-1]

    else:
        pass_margin = bid_buyer_pool.shape[0]
        breakeven_index = np.where(bid_buyer_pool.iloc[:, 0].values >= bid_seller_pool.iloc[:pass_margin, 0].values)[0][-1]

    if auction_mechanism == 'VCG':
        buyer_breakeven_price = bid_buyer_pool.iloc[breakeven_index, 0]
        seller_breakeven_price = bid_seller_pool.iloc[breakeven_index, 0]

        if breakeven_index + 1 == bid_buyer_pool.shape[0]:
            buyer_next_price = bid_buyer_pool.iloc[breakeven_index, 0]
        else:
            buyer_next_price = bid_buyer_pool.iloc[breakeven_index + 1, 0]

        if breakeven_index + 1 == bid_seller_pool.shape[0]:
            seller_next_price = bid_seller_pool.iloc[breakeven_index, 0]
        else:
            seller_next_price = bid_seller_pool.iloc[breakeven_index + 1, 0]

        buyer_market_price = max(seller_breakeven_price, buyer_next_price)
        seller_market_prcie = min(seller_next_price, buyer_breakeven_price)

    return buyer_market_price, seller_market_prcie, breakeven_index



def market_clearing(options: ProbOptions, IFN: ReadInputData, ST: SetTimes, UI: UserInfo):
    """
    users 옥션 결과 book frame 생성 및 초기값 입력
    """
    users_book = dict()
    for d, h_resolution in ST.date_list_resol:
        open_day = (ST.first_day + timedelta(days=d - 1)).strftime("%m-%d")
        date_resolution = f"{open_day}-{h_resolution}"
        users_book[date_resolution] = dict()
        for u in UI.Users.keys():
            users_book[date_resolution][u] = \
                {
                    'P_winning': [],
                    'Q_winning': [],
                    'cummul_Q': 0
                }

    """
    옥션 시작
    """
    auction_book = dict()
    date_list = []
    for d, h_resolution in ST.date_list_resol:
        open_day = (ST.first_day + timedelta(days=d-1)).strftime("%m-%d")
        date_resolution = f"{open_day}-{h_resolution}"
        date_list.append(date_resolution)

        """
        거래 시간 별 및 거래 참여자 별 입찰서 book 생성
        """
        # auction 개장 "일자 - 거래 시점"에 대한 book 생성
        auction_book[date_resolution] = dict()

        # 시장 평균 가격 (평균 SMP)
        market_price = np.mean(IFN.smp_data.loc[open_day, [f"{h}h" for h in ST.trading_time[h_resolution]]])

        # auction book으로 거래시점 별 평균 시장 가격 입력
        auction_book[date_resolution]['market-price'] = market_price
        auction_book[date_resolution]['buyers'] = dict()
        auction_book[date_resolution]['sellers'] = dict()

        """
        옥션 참가자(buyer, seller)의 각 입찰량 및 입찰상한가격, 입찰하한가격 결정 
        """
        for user in UI.Users.keys():
            quantity = \
                int(
                    UI.Users[user]['Electricity_Consumption'][d, h_resolution] -
                    UI.Users[user]['Generation'][d, h_resolution]
                )


            # resolution된 시간대의 netload의 합으로 판매량 결정
            # 동일 resolution에서는 자신의 전력량을 낮춘다고 가정
            # 즉, 공급과잉량에 대해서 판매자는 시장가격으로 정산을 받거나 또는 구매자에게 파는 것만을 고려해야 함.
            # 자신의 누진단계를 낮추도록 할 수 없음. (즉, 상계거래 폐지)

            if quantity < 0:
                auction_book[date_resolution]['sellers'][user] = \
                    {
                        'quantity': quantity,
                        'upper_price': ST.highest_block_price,
                        'lower_price': market_price
                    }

            else:

                # 낙찰량(구매자에서 낙찰된 양)을 뺀 순소비량에 대해서 누진단계 결정
                block_name = \
                    _set_block_step(
                        net_ToC=
                        sum(
                            UI.Users[user]['Net_consumption'].values()
                        ) -
                        sum(
                            [users_book[date][user]['cummul_Q']
                             for date in date_list
                             if users_book[date][user]['cummul_Q'] > 0]
                        ),
                        tariff_season=ST.tariff_season
                    )

                upper_price = ST.tariff_season.loc[block_name, [f"{UI.Users[user]['Vol_level']}, 전력량"]].iloc[0]

                if upper_price >= market_price:
                    lower_price = market_price
                else:
                    lower_price = upper_price

                auction_book[date_resolution]['buyers'][user] = \
                    {
                        'quantity': quantity,
                        'upper_price': upper_price,
                        'lower_price': lower_price
                    }


        # 거래량 산출
        trading_amount = \
            abs(
                sum(
                    [auction_book[date_resolution]['sellers'][u]['quantity']
                     for u in auction_book[date_resolution]['sellers'].keys()]
                )
            )
        # 판매 물량이 없을 경우, 거래 종료
        if trading_amount == 0:
            auction_book[date_resolution]['clearing-price'] = 0

        else:
            # 임시의 입찰 정보 --> (가변함)
            buyer_info_temp = copy.deepcopy(auction_book[date_resolution]['buyers'])
            seller_info_temp = copy.deepcopy(auction_book[date_resolution]['sellers'])
            buyer_clearing_price = None
            seller_clearing_price = None
            for open_num in range(options.number_of_market_open):

                # 입찰 전략에 따른 입찰가 생성
                bid_buyer_pool = _bid_by_strategy(
                    options=options,
                    player_info=buyer_info_temp,
                    strategy=options.bidding_strategy,
                    clearing_price=buyer_clearing_price
                )

                bid_seller_pool = _bid_by_strategy(
                    options=options,
                    player_info=seller_info_temp,
                    strategy=options.bidding_strategy,
                    clearing_price=seller_clearing_price
                )

                if bid_buyer_pool.shape[0] == 0 or bid_seller_pool.shape[0] == 0:
                    break

                # natural ordering을 통한 입찰가 정렬 (buyer: descending order, seller: descending order)
                bid_buyer_pool = \
                    bid_buyer_pool.sort_values('bid', ascending=False)

                bid_seller_pool = \
                    bid_seller_pool.sort_values('bid', ascending=True)

                # market clearing 방식에 따른 market price 결정
                try:
                    buyer_clearing_price, seller_clearing_price, breakeven_index = \
                        _market_price(
                            options.auction_mechanism,
                            bid_buyer_pool,
                            bid_seller_pool
                        )
                # 구매자 또는 판매자의 모든 입찰가가 counterpart의 입찰가와 매칭되지 않을 경우
                except IndexError:
                    break

                """
                낙찰된 구매자 및 판매자의 정산
                """
                # buyer-Market-clearing [결정된 시장 가격에 따른 clearing 단계]
                winning_player = []

                bid_buyer_pool['name'] = \
                    [x.split(',')[0] for x in list(bid_buyer_pool.index)]
                buyer_count_df = bid_buyer_pool.iloc[:breakeven_index+1, :].groupby('name').count()
                for user in buyer_count_df.index:
                    users_book[date_resolution][user]['Q_winning'].append(
                        buyer_count_df.loc[user, 'bid']
                    )
                    users_book[date_resolution][user]['P_winning'].append(
                        buyer_count_df.loc[user, 'bid'] * buyer_clearing_price
                    )
                    users_book[date_resolution][user]['cummul_Q'] = \
                        sum(users_book[date_resolution][user]['Q_winning'])
                    winning_player.append(user)

                # seller-Market-clearing [결정된 시장 가격에 따른 clearing 단계]
                bid_seller_pool['name'] = \
                    [x.split(',')[0] for x in list(bid_seller_pool.index)]
                seller_count_df = bid_seller_pool.iloc[:breakeven_index+1, :].groupby('name').count()
                for user in seller_count_df.index:
                    users_book[date_resolution][user]['Q_winning'].append(
                        - seller_count_df.loc[user, 'bid']
                    )
                    users_book[date_resolution][user]['P_winning'].append(
                        - seller_count_df.loc[user, 'bid'] * seller_clearing_price
                    )
                    users_book[date_resolution][user]['cummul_Q'] = \
                        sum(users_book[date_resolution][user]['Q_winning'])
                    winning_player.append(user)

                for user in list(set(UI.Users.keys()) - set(winning_player)):
                    users_book[date_resolution][user]['Q_winning'].append(
                        0
                    )
                    users_book[date_resolution][user]['P_winning'].append(
                        0
                    )
                    users_book[date_resolution][user]['cummul_Q'] = \
                        sum(users_book[date_resolution][user]['Q_winning'])

                """
                이전 market에 입찰가 중 평균 입찰가를 historic bid로 결정
                """
                for user in buyer_info_temp.keys():
                    buyer_info_temp[user]['quantity'] = \
                        auction_book[date_resolution]['buyers'][user]['quantity'] - \
                        users_book[date_resolution][user]['cummul_Q']
                    buyer_info_temp[user]['historic_bid'] = \
                        bid_buyer_pool[bid_buyer_pool.name == user]['bid'].mean()
                for user in seller_info_temp.keys():
                    seller_info_temp[user]['quantity'] = \
                        auction_book[date_resolution]['sellers'][user]['quantity'] - \
                        users_book[date_resolution][user]['cummul_Q']
                    seller_info_temp[user]['historic_bid'] = \
                        bid_seller_pool[bid_seller_pool.name == user]['bid'].mean()

    return auction_book, users_book