import random
import numpy as np
from datetime import timedelta, datetime

import lib.parameters as para


class Auction:
    def __init__(self, options: para.ProbOptions, IFN: para.ReadInputData, ST: para.SetTimes, UI: para.UserInfo):
        self.highest_block_price = IFN.tariff_table[IFN.tariff_table['비고'] == options.season].loc[:, ['저압, 전력량', '고압, 전력량']].max().max()

        # 결과 book 초기값 설정
        self.result_book = dict()
        for d in range(ST.first_day.day, ST.last_day.day + 1):
            for h_resolution in ST.trading_time.keys():
                self.result_book[f"{d}-{h_resolution}"] = dict()
                for u in UI.Users.keys():
                    self.result_book[f"{d}-{h_resolution}"][u] = self.set_result_book(0, 0)

        self.auction_book = dict()
        self.pre_market_order = dict()
        for d in range((ST.last_day - ST.first_day).days + 1):
            open_day = (ST.first_day + timedelta(days=d)).strftime("%m-%d")
            for h_resolution in ST.trading_time.keys():
                self.auction_book[f"{open_day}-{h_resolution}"] = dict()    # auction 개장 날짜에 대한 pool 생성

                self.market_rates = np.mean(IFN.smp_data.loc[open_day, [f"{h}h" for h in ST.trading_time[h_resolution]]])
                self.auction_book[f"{open_day}-{h_resolution}"]['market rates'] = self.market_rates     # auction 날짜에 대한 시장 평균 가격 넣기

                # self.auction_book[f"{open_day}-{h_resolution}"]['buyer clearing price'] = 0
                # self.auction_book[f"{open_day}-{h_resolution}"]['seller clearing price'] = 0
                self.market_book = dict()
                self.market_book['buyers'] = dict()
                self.market_book['sellers'] = dict()

                self.divide_agent(open_day, h_resolution, options, UI, ST, IFN)    # User를 구매자, 판매자로 분류하여 bidding 크기와 bidding price의 상한선을 agent 별로 설정

                # auction 시작
                trading_amount = abs(sum([self.market_book['sellers'][k]['quantity'] for k in self.market_book['sellers'].keys()]))
                if trading_amount == 0:
                    self.auction_book[f"{open_day}-{h_resolution}"]['clearing price'] = 0
                else:
                    for i in range(trading_amount):
                        buyers = [b for b in self.market_book['buyers'] if self.market_book['buyers'][b]['quantity'] > 0]
                        sellers = [s for s in self.market_book['sellers'] if self.market_book['sellers'][s]['quantity'] < 0]

                        orders_of_buyer = []
                        orders_of_seller = []
                        for b in random.sample(buyers, len(buyers)):
                            orders_of_buyer.append(self.set_buyer_order_book(open_day=open_day,
                                                                             h_resolution=h_resolution,
                                                                             options=options,
                                                                             user=b,
                                                                             bidding_Q=1,
                                                                             bidding_T=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                                                             )
                                                   )
            # randomly 하게 bidding price를 agent별로 설정. 단, historic winning price가 있을 경우, 해당 값을 이용하여 bidding price 결정
                # winning bidding이 없을 경우, 최대 5번까지 auction 수행
                # natural ordering 수행
                # clearing price 결정 및 정산 수행

    def extract_positive_margin(self):
        while True:
            margin = np.random.normal(0.2, 0.15)
            if margin > 0:
                break
            else:
                pass
        return margin

    def set_buyer_order_book(self, open_day, h_resolution, options: para.ProbOptions, user, bidding_Q, bidding_T):
        if options.bidding_strategy == 'Randomly':
            bidding_P = np.random.uniform(self.market_book['buyers'][user]['lower price'],
                                  self.market_book['buyers'][user]['upper price'])
        elif options.bidding_strategy == 'Update margin':
            if (open_day == '01-01' and h_resolution == 1) or pre_market_order == dict():
                bidding_P = np.random.uniform(self.market_book['buyers'][user]['lower price'],
                                      self.market_book['buyers'][user]['upper price'])
            else:
                margin = self.extract_positive_margin()
                if pre_market_order['buyers'][user]['bidding price'] < pre_market_order['clearing price']:  # 이전 옥션에서 낙찰됨
                    bidding_P = pre_market_order['buyers'][user]['bidding price'] * (1 - margin)
                    if bidding_P < self.market_book['buyers'][user]['lower price']:
                        bidding_P = self.market_book['buyers'][user]['lower price']
                    else:
                        pass
                else:   # 이전 옥션에서 낙찰받지 못함
                    bidding_P = pre_market_order['buyers'][user]['bidding price'] * (1 + margin)
                    if bidding_P > self.market_book['buyers'][user]['upper price']:
                        bidding_P = self.market_book['buyers'][user]['upper price']
                    else:
                        pass

        return {'name': user,
                'bidding_P': bidding_P,
                'bidding_Q': bidding_Q,
                'bidding_T': bidding_T
                }


    def set_result_book(self,P, Q):
        return {'P_winning': P,
                'Q_winning': Q
                }

    # def set_Market_book(self, open_day, h_resolution, ST):
    #     return {'open day-hours': [f"{open_day}-{h}" for h in ST.trading_time[h_resolution]],
    #             'clearing price': 0,
    #             'buyers': {},
    #             'sellers': {}
    #             }

    def set_PQ_book(self, q, upper_p, lower_p):
        return {'quantity': q,
                'upper price': upper_p,
                'lower price': lower_p
                }

    def set_vol_level(self, user, UI: para.UserInfo):
        contract_power = max(UI.Users[user]['Electricity_Consumption'].values())
        if contract_power <= 3:
            vol_level = '저압'
        else:
            vol_level = '고압'
        return vol_level

    def set_block_step(self, user, UI: para.UserInfo, options: para.ProbOptions, IFN: para.ReadInputData):
        amount = UI.Users[user]['Total_amount_of_consumption'] - \
                 sum([self.result_book[h][user]['Q_winning'] for h in self.result_book.keys() for u in self.result_book[h].keys() if u == user])
        tariff_season = IFN.tariff_table[IFN.tariff_table['비고'] == options.season]
        tariff_block = tariff_season[amount < tariff_season.loc[:, "max"]].iloc[0, :]
        return tariff_block.name

    def divide_agent(self, open_day, h_resolution, options: para.ProbOptions, UI: para.UserInfo, ST: para.SetTimes, IFN: para.ReadInputData):
        for user in UI.Users.keys():
            if UI.Users[user]['Gen_bin'] == 1:
                netload = [v - UI.Users[user]['Generation'][k]
                           for k, v in UI.Users[user]['Electricity_Consumption'].items()
                           if k[0] == int(open_day.split('-')[1]) and k[1] in ST.trading_time[h_resolution]]
            else:
                netload = [v
                           for k, v in UI.Users[user]['Electricity_Consumption'].items()
                           if k[0] == int(open_day.split('-')[1]) and k[1] in ST.trading_time[h_resolution]]

            excess_E = int(sum([k for k in netload if k < 0]))      # 소수점 이하 절사
            if excess_E < 0:
                self.market_book['sellers'][user] = self.set_PQ_book(excess_E,
                                                                     self.highest_block_price,
                                                                     self.market_rates
                                                                     )
            else:
                vol_level = self.set_vol_level(user, UI)
                block_name = self.set_block_step(user, UI, options, IFN)
                upper_P = IFN.tariff_table[IFN.tariff_table['비고'] == options.season].loc[block_name, [f"{vol_level}, 전력량"]].iloc[0]
                if upper_P >= self.market_rates:
                    lower_P = self.market_rates
                else:
                    lower_P = upper_P
                self.market_book['buyers'][user] = self.set_PQ_book(int(sum([k for k in netload if k > 0])),
                                                                    upper_P,
                                                                    lower_P
                                                                    )




    # def classify_users(self, trading_day, trading_time, Users):
