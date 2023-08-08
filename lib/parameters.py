import pandas as pd
import numpy as np
import json
import random

from datetime import datetime, timedelta
from dateutil import relativedelta


class ProbOptions:
    def __init__(self, input_data_name=None):
        if input_data_name is None:
            self.input_data = None
            self.Number_Of_Users = 100
            self.Number_Of_Consumers = 50
            self.Number_Of_Prosumers = 50
        else:
            self.input_data = pd.read_csv("{n}.csv".format(n=input_data_name), index_col=0)
            self.Number_Of_Users = self.input_data.index.shape[0]
            self.Number_Of_Prosumers = self.input_data[self.input_data.capacity > 0].shape[0]
            self.Number_Of_Consumers = self.input_data[self.input_data.capacity == 0].shape[0]


        self.year = 2021
        self.month = 1

        self.demand_noise_dist_deviation_ratio = 0.3
        self.trading_time_option = 'Time-of-Use'

        self.payback_year = 10
        """
        
        Vickrey-Clark-Groves(VCG)
        """
        self.auction_mechanism = 'VCG'

        # 거래 시점별 거래 반복 횟수
        self.number_of_market_open = 100
        """
        입찰 전략 선택
        1. Randomly
        2. Update margin
        """
        self.bidding_strategy = 'Update margin'

        self.loc_data = f"data"

        self.margin_rate_avr = 0.2
        self.margin_rate_sd = 0.15
    def extract_margin_rate(self):
        return abs(np.random.normal(0.2, 0.15))


class ReadInputData:
    def __init__(self, options: ProbOptions):
        self.demand_factor = pd.read_excel(f"{options.loc_data}/demand.xlsx", sheet_name='주택용', index_col=0)
        self.consumption_mean, self.consumption_std = pd.read_excel(f"{options.loc_data}/electricity_consumption_normal_parameter.xlsx",
                                                                    index_col=0).loc[options.month, :].values

        # with open(f"{options.loc_data}/excess_electricity_sample_set.json", "r") as f:
        #     excess_electricity_sample_set = json.load(f)
        # self.excess_electricity_sample_set = excess_electricity_sample_set[f"{options.month}월"]

        with open(f"{options.loc_data}/pv_gen_sample_set.json", "r") as f:
            self.pv_gen_sample_set = json.load(f)

        # self.self_pv_sample_set = pd.read_excel(f"{options.loc_data}/self_generation_sample_set.xlsx").values
        pv_cap_historic = pd.read_excel(f"{options.loc_data}/용량 별 태양광 용량.xlsx", sheet_name='데이터', index_col=0, header=0)
        pv_cap_sample_set = []
        pv_cap_sample_set.append(np.random.uniform(0, 1, int(pv_cap_historic.loc['1kW 이하', '2020'] / ((0 + 1) / 2))).tolist())
        pv_cap_sample_set.append(np.random.uniform(1, 3, int(pv_cap_historic.loc['1~3kW 이하', '2020'] / ((1 + 3) / 2))).tolist())
        pv_cap_sample_set.append(np.random.uniform(3, 10, int(pv_cap_historic.loc['3~10kW 이하', '2020'] / ((3 + 10) / 2))).tolist())
        pv_cap_sample_set.append(np.random.uniform(10, 50, int(pv_cap_historic.loc['10~50kW 이하', '2020'] / ((10 + 50) / 2))).tolist())
        pv_cap_sample_set.append(np.random.uniform(50, 100, int(pv_cap_historic.loc['50~100kW 이하', '2020'] / ((50 + 100) / 2))).tolist())
        pv_cap_sample_set.append(np.random.uniform(100, 500, int(pv_cap_historic.loc['100~500kW 이하', '2020'] / ((100 + 500) / 2))).tolist())
        pv_cap_sample_set.append(np.random.uniform(500, 1000, int(pv_cap_historic.loc['500~1,000kW 이하', '2020'] / ((500 + 1000) / 2))).tolist())

        self.pv_cap_sample_set = sum(pv_cap_sample_set, [])

        self.tariff_table = pd.read_excel(f"{options.loc_data}/tariff_table.xlsx", sheet_name='주택용(22.10.01)', index_col=0)

        smp_data = pd.read_excel(f"{options.loc_data}/smp_land_2019.xlsx", sheet_name='smp_land_2019', index_col=0, header=1)
        smp_index = [f"{k.replace('2019', '')[0:2]}-{k.replace('2019', '')[2:]}" for k in smp_data.index.astype('str')]
        self.smp_data = pd.DataFrame(smp_data.values, index=smp_index, columns=smp_data.columns)

        self.PF_mean, self.PF_sigma = 0.2, 0.15

        generation_pat = self.pv_gen_sample_set

        ind_ = range(1, 366, 1)
        col_ = range(1, 25, 1)

        self.Gen_prob_pattern = pd.DataFrame(index=ind_, columns=col_)
        for i in ind_:
            for j in col_:
                mean_ = np.mean(generation_pat[f"{i-1},{j-1}"])
                std_ = np.std(generation_pat[f"{i-1},{j-1}"])
                while True:
                    val_ = np.random.normal(loc=mean_, scale=std_)
                    if val_ >= 0:
                        break
                    else:
                        pass
                self.Gen_prob_pattern.loc[i, j] = val_

        self.PV_unit_price = 1788939


class SetTimes:

    def _set_trading_time(self, options):
        trading_time_dict = dict()
        if options.trading_time_option == 'Time-of-Use':
            if options.month in [1, 2, 11, 12]:
                trading_time_dict[1] = [1, 2, 3, 4, 5, 6, 7, 8, 23, 24]
                trading_time_dict[2] = [9, 13, 14, 15, 16, 20, 21, 22]
                trading_time_dict[3] = [10, 11, 12, 17, 18, 19]
            else:
                trading_time_dict[1] = [1, 2, 3, 4, 5, 6, 7, 8, 23, 24]
                trading_time_dict[2] = [9, 10, 11, 13, 19, 20, 21, 22]
                trading_time_dict[3] = [12, 14, 15, 16, 17, 18]
        self.trading_time = trading_time_dict

    def _set_season(self, options:ProbOptions):
        if options.month == 7 or options.month == 8:
            return '하계'
        else:
            return '기타'

    def main(self, options:ProbOptions, IFN: ReadInputData):
        this_month = datetime(year=options.year, month=options.month, day=1).date()
        next_month = this_month + relativedelta.relativedelta(months=1)
        self.first_day = this_month
        self.last_day = next_month - timedelta(days=1)

        self._set_trading_time(options)
        self.date_list = [(d, h) for d in range(self.first_day.day, self.last_day.day + 1)
                          for h in range(1, 24 + 1)]
        self.date_list_resol = [(d, h_resolution) for d in range(self.first_day.day, self.last_day.day + 1)
                                for h_resolution in self.trading_time.keys()]
        self.season = self._set_season(options)
        self.tariff_season = IFN.tariff_table[IFN.tariff_table['비고'] == self.season]
        self.highest_block_price = \
            IFN.tariff_table[IFN.tariff_table['비고'] == self.season].loc[:, ['저압, 전력량', '고압, 전력량']].max().max()

        self.date_list_resol_mdh = list()
        for d, h_resolution in self.date_list_resol:
            open_day = (self.first_day + timedelta(days=d - 1)).strftime("%m-%d")
            self.date_list_resol_mdh.append(f"{open_day}-{h_resolution}")


class UserInfo:
    def __init__(self, options: ProbOptions):
        self.items = [
            'Gen_bin',
            'Electricity_Consumption',
            'Generation',
            'Feed_in',
            'Net_consumption',
            'Block_step_old',
            'Demand_charge',
            'Energy_charge',
            'Installed_Cap',
            'Min_generation_price',
            'Vol_level'
        ]


        self.auction_book = dict()
        self.users_book = dict()
        self.res_book = dict()
        # Users SET 만들기
        self.make_users_set(options)

    def make_users_set(self, options:ProbOptions):
        Users = dict()
        if options.input_data is None:
            users_list = list(range(1, options.Number_Of_Users + 1, 1))
            for i in range(options.Number_Of_Consumers):
                b = random.choice(users_list)
                users_list.pop(users_list.index(b))
                Users[f"u{b}"] = Users.fromkeys(self.items, 0)
                Users[f"u{b}"]['Gen_bin'] = 0

            for i in range(options.Number_Of_Prosumers):
                p = random.choice(users_list)
                users_list.pop(users_list.index(p))
                Users[f"u{p}"] = Users.fromkeys(self.items, 0)
                Users[f"u{p}"]['Gen_bin'] = 1
        else:
            for u in options.input_data.index:
                cap = options.input_data.loc[u, "capacity"]
                Users[f"u{u}"] = Users.fromkeys(self.items, 0)
                if cap > 0:
                    Users[f"u{u}"]['Gen_bin'] = 1
                else:
                    Users[f"u{u}"]['Gen_bin'] = 0

        Users_sorted = dict()
        for u in sorted(Users.items(), key=lambda item: int(item[0][1:])):
            Users_sorted[u[0]] = u[1]

        self.Users = Users_sorted

    def put_consumption_to_Users(self, options:ProbOptions, IFN:ReadInputData, ST:SetTimes):
        for u in self.Users.keys():
            demand_date_factor = self.make_random_demand_factor(options, IFN)
            consumption_pattern = self.make_electricity_consumption_pattern_by_user(options, IFN, demand_date_factor, u)

            # prosumer의 계약전력은 순소비패턴이 아닌 실제 소비패턴으로 결정
            contract_voltage = max(consumption_pattern.values())
            if contract_voltage <= 3:
                vol_level = '저압'
            else:
                vol_level = '고압'
            self.Users[u]['Vol_level'] = vol_level

            consumption_pattern_resol = dict()
            for d, h in ST.date_list:
                consumption_pattern_resol[d, h] = float(consumption_pattern[d, h])

            self.Users[u]['Electricity_Consumption'] = consumption_pattern_resol

    def put_generation_to_Users(self, options:ProbOptions, IFN:ReadInputData, ST:SetTimes):
        gen_prob_pattern_dict = self.make_gen_pattern_dict(IFN.Gen_prob_pattern)
        for u in self.Users.keys():
            if self.Users[u]['Gen_bin'] == 1:
                if options.input_data is None:
                    while True:
                        # cap_rand = np.round(np.random.normal(loc=IFN.pv_cap_mean, scale=IFN.pv_cap_std), 1)
                        _cap = np.round(random.choice(IFN.pv_cap_sample_set), 2)
                        if _cap >= 0:
                            break
                        else:
                            pass
                else:
                    _cap = options.input_data.loc[int(u[1:]), "capacity"]

                self.Users[u]['Installed_Cap'] = float(_cap)
                self.Users[u]['Min_generation_price'] = \
                    round(IFN.PV_unit_price / (IFN.Gen_prob_pattern.sum().sum() * options.payback_year))
            else:
                _cap = 0
                self.Users[u]['Installed_Cap'] = 0

            generation_resol = dict()
            for d, h in ST.date_list:
                generation_resol[d, h] = float(gen_prob_pattern_dict[d, h] * _cap)

            self.Users[u]['Generation'] = generation_resol

    def put_NetInfo_to_Users(self, ST:SetTimes):
        for u in self.Users.keys():
            self.Users[u]['Feed_in'] = dict()
            self.Users[u]['Net_consumption'] = dict()
            for date in ST.date_list:
                if self.Users[u]['Gen_bin'] == 1:
                    if self.Users[u]['Electricity_Consumption'][date] >= self.Users[u]['Generation'][date]:
                        self.Users[u]['Feed_in'][date] = 0
                        self.Users[u]['Net_consumption'][date] = self.Users[u]['Electricity_Consumption'][date] - \
                                                                 self.Users[u]['Generation'][date]
                    else:
                        self.Users[u]['Feed_in'][date] = self.Users[u]['Generation'][date] - \
                                                         self.Users[u]['Electricity_Consumption'][date]
                        self.Users[u]['Net_consumption'][date] = 0
                else:
                    self.Users[u]['Feed_in'][date] = 0
                    self.Users[u]['Net_consumption'][date] = self.Users[u]['Electricity_Consumption'][date]

    def put_TariffInfo_to_Users(self, options, IFN, ST):
        for u in self.Users.keys():
            _vol = f"{self.Users[u]['Vol_level']}"
            if self.Users[u]['Gen_bin'] == 1:
                amount = sum(self.Users[u]['Net_consumption'].values())
            else:
                amount = sum(self.Users[u]['Electricity_Consumption'].values())

            block = ST.tariff_season[amount < ST.tariff_season.loc[:, "max"]].iloc[0, :]
            step = block.name
            demand_charge = block[f"{_vol}, 기본"]

            block_lower = ST.tariff_season[amount >= ST.tariff_season.loc[:, "max"]]
            energy_charge = 0
            max_val = 0
            for i in block_lower.index:
                energy_charge += (block_lower.loc[i, 'max'] - max_val) * block_lower.loc[i, f"{_vol}, 전력량"]
                max_val = block_lower.loc[i, 'max']

            energy_charge += (amount - max_val) * block[f"{_vol}, 전력량"]

            self.Users[u]['Block_step_old'] = int(step)
            self.Users[u]['Demand_charge'] = int(demand_charge)
            self.Users[u]['Energy_charge'] = float(energy_charge)


    def make_gen_pattern_dict(self, Gen_prob_pattern):
        gen_prob_pattern_dict = dict()
        for d in range(self.first_day.day, self.last_day.day + 1):
            for h in range(1, 25):
                gen_prob_pattern_dict[d, h] = float(Gen_prob_pattern.loc[d, h])

        return gen_prob_pattern_dict

    def make_random_demand_factor(self, options:ProbOptions, IFN:ReadInputData):
        def what_day_is_it(date):
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            day = date.weekday()
            return days[day]

        Demand_date_factor = dict()
        for d in range(self.first_day.day, self.last_day.day + 1):
            day = what_day_is_it(date=datetime(options.year, options.month, d).date())
            for h in range(1, 25):
                if day in ['Mon', 'Sat', 'Sun']:
                    pass
                else:
                    day = 'Working'
                mean = float(IFN.demand_factor.loc[f"{options.month},{h}", day])
                Demand_date_factor[d, h] = np.random.normal(loc=mean,
                                                                 scale=mean * options.demand_noise_dist_deviation_ratio
                                                                 )

        max_value = max([v for k, v in Demand_date_factor.items()])
        for k, v in Demand_date_factor.items():
            Demand_date_factor[k] = Demand_date_factor[k] / max_value

        return Demand_date_factor

    def make_electricity_consumption_pattern_by_user(self, options:ProbOptions, IFN, demand_date_factor, user):
        if options.input_data is None:
            while True:
                electricity_amount = np.random.normal(IFN.consumption_mean, IFN.consumption_std)
                if electricity_amount > 0:
                    break
                else:
                    pass
        else:
            electricity_amount = options.input_data.loc[int(user[1:]), str(options.month)]

        consumption_pattern = dict()
        factor_sum = sum(demand_date_factor.values())
        for k, v in demand_date_factor.items():

            consumption_pattern[k] = demand_date_factor[k] * (electricity_amount / factor_sum)

        return consumption_pattern

    def main(self, options:ProbOptions, IFN:ReadInputData, ST:SetTimes):
        self.first_day = ST.first_day
        self.last_day = ST.last_day

        self.put_consumption_to_Users(options, IFN, ST)
        self.put_generation_to_Users(options, IFN, ST)
        self.put_NetInfo_to_Users(ST)
        self.put_TariffInfo_to_Users(options, IFN, ST)

    def res_info(self, options:ProbOptions, ST:SetTimes):
        _column = [
            "순 조달량",
            "누진단계",
            "역송량",
            "판매량",
            "잔여 크레딧",
            "구매 낙찰량",
            "구매 낙찰가격",
            "판매 낙찰량",
            "판매 낙찰가격",
            "미 판매량",
            "월간 평균 SMP",
            "상계거래 후 잔여크레딧 보상요금",
            "옥션 후 SMP 보상요금",
            "기본요금",
            "옥션 후 순 조달량",
            "옥션 후 누진단계",
            "옥션 후 전력량요금",
            "상계거래 후 순 조달량",
            "상계거래 후 누진단계",
            "상계거래 후 전력량요금",
            "옥션 후 총 비용",
            "상계거래 후 총 비용",
            "편익"
        ]

        _index = self.Users.keys()

        info_df = pd.DataFrame(index=_index, columns=_column)

        for u in _index:
            info_df.at[u, "순 조달량"] = sum(self.Users[u]["Net_consumption"].values())
            info_df.at[u, "누진단계"] = self.Users[u]["Block_step_old"]
            info_df.at[u, "역송량"] = sum(self.Users[u]['Feed_in'].values())
            info_df.at[u, "판매량"] = - sum([int(v) for v in self.Users[u]["Feed_in"].values()])

            amount = info_df.at[u, "순 조달량"] - info_df.at[u, "역송량"]
            if amount >= 0:
                info_df.at[u, "잔여 크레딧"] = 0

            else:
                info_df.at[u, "잔여 크레딧"] = -amount

            info_df.at[u, "구매 낙찰량"] = sum(
                [
                    sum(self.users_book[date][u]["Q_winning"])
                    for date in ST.date_list_resol_mdh
                    if u in self.auction_book[date]["buyers"].keys() and sum(self.users_book[date][u]["Q_winning"]) > 0
                ]

            )

            info_df.at[u, "구매 낙찰가격"] = sum(
                [
                    sum(
                        self.users_book[date][u]["P_winning"]
                    )
                    for date in ST.date_list_resol_mdh
                    if u in self.auction_book[date]["buyers"].keys() and sum(self.users_book[date][u]["Q_winning"]) > 0
                ]
            )

            info_df.at[u, "판매 낙찰량"] = sum(
                [
                    sum(self.users_book[date][u]["Q_winning"])
                    for date in ST.date_list_resol_mdh
                    if u in self.auction_book[date]["sellers"].keys() and sum(self.users_book[date][u]["Q_winning"]) < 0
                ]
            )

            info_df.at[u, "판매 낙찰가격"] = \
                sum([
                    sum(
                        self.users_book[date][u]["P_winning"]
                    )
                    for date in ST.date_list_resol_mdh
                    if u in self.auction_book[date]["sellers"].keys() and sum(self.users_book[date][u]["Q_winning"]) < 0
                ]
                )

            info_df.at[u, '미 판매량'] = info_df.at[u, '역송량'] + info_df.at[u, "판매 낙찰량"]

            info_df.at[u, "월간 평균 SMP"] = np.average(
                [
                    self.auction_book[date]["market-price"]
                    for date in ST.date_list_resol_mdh
                ]
            )

            info_df.at[u, "상계거래 후 잔여크레딧 보상요금"] = \
                - info_df.at[u, "잔여 크레딧"] * info_df.at[u, "월간 평균 SMP"]

            # 미낙찰량에 대한 실시간 SMP 기준 보상
            info_df.at[u, "옥션 후 SMP 보상요금"] = - sum(
                [
                    self.auction_book[date]["market-price"] *
                    (
                            sum(self.Users[u]["Feed_in"][int(date.split('-')[1]), _h] for _h in ST.trading_time[int(date.split('-')[2])]) +
                            sum(self.users_book[date][u]["Q_winning"])
                    )
                    for date in ST.date_list_resol_mdh
                    if self.Users[u]["Gen_bin"] == 1
                ]
            )

            info_df.at[u, "기본요금"] = self.Users[u]["Demand_charge"]

            info_df.at[u, "옥션 후 순 조달량"] = \
                sum(self.Users[u]["Net_consumption"].values()) - info_df.at[u, "구매 낙찰량"]

            info_df.at[u, "옥션 후 누진단계"] = \
                ST.tariff_season[info_df.at[u, "옥션 후 순 조달량"] < ST.tariff_season.loc[:, "max"]].iloc[0, :].name

            block = ST.tariff_season[info_df.at[u, "옥션 후 순 조달량"] < ST.tariff_season.loc[:, "max"]].iloc[0, :]
            block_lower = ST.tariff_season[info_df.at[u, "옥션 후 순 조달량"] >= ST.tariff_season.loc[:, "max"]]
            energy_charge = 0
            max_val = 0
            for i in block_lower.index:
                energy_charge += \
                    (block_lower.loc[i, 'max'] - max_val) * \
                    block_lower.loc[i, f"{self.Users[u]['Vol_level']}, 전력량"]
                max_val = block_lower.loc[i, 'max']

            energy_charge += \
                (info_df.at[u, "옥션 후 순 조달량"] - max_val) * block[f"{self.Users[u]['Vol_level']}, 전력량"]

            info_df.at[u, "옥션 후 전력량요금"] = energy_charge

            net_ToC_nm = sum(self.Users[u]["Net_consumption"].values()) - sum(self.Users[u]["Feed_in"].values())
            if net_ToC_nm > 0:
                info_df.at[u, "상계거래 후 순 조달량"] = net_ToC_nm
            else:
                info_df.at[u, "상계거래 후 순 조달량"] = 0

            info_df.at[u, "상계거래 후 누진단계"] = \
                ST.tariff_season[info_df.at[u, "상계거래 후 순 조달량"] < ST.tariff_season.loc[:, "max"]].iloc[0, :].name

            block = ST.tariff_season[info_df.at[u, "상계거래 후 순 조달량"] < ST.tariff_season.loc[:, "max"]].iloc[0, :]
            block_lower = ST.tariff_season[info_df.at[u, "상계거래 후 순 조달량"] >= ST.tariff_season.loc[:, "max"]]
            energy_charge = 0
            max_val = 0
            for i in block_lower.index:
                energy_charge += \
                    (block_lower.loc[i, 'max'] - max_val) * \
                    block_lower.loc[i, f"{self.Users[u]['Vol_level']}, 전력량"]
                max_val = block_lower.loc[i, 'max']

            energy_charge += \
                (info_df.at[u, "상계거래 후 순 조달량"] - max_val) * block[f"{self.Users[u]['Vol_level']}, 전력량"]

            info_df.at[u, "상계거래 후 전력량요금"] = energy_charge

            info_df.at[u, "옥션 후 총 비용"] = \
                info_df.at[u, "구매 낙찰가격"] + info_df.at[u, "판매 낙찰가격"] + info_df.at[u, "옥션 후 SMP 보상요금"] + \
                info_df.at[u, "기본요금"] + info_df.at[u, "옥션 후 전력량요금"]

            info_df.at[u, "상계거래 후 총 비용"] = \
                info_df.at[u, "상계거래 후 잔여크레딧 보상요금"] + info_df.at[u, "기본요금"] + info_df.at[u, "상계거래 후 전력량요금"]

            info_df.at[u, "편익"] = \
                info_df.at[u, "상계거래 후 총 비용"] - info_df.at[u, "옥션 후 총 비용"]

        self.res_book[options.month] = info_df

    def save_res(self, month=None, typ=None, name=None):
        """

        month :
        typ :
            1 : 개별 저장
            2 : 모든 월을 한번에 저장
        """
        if typ == 1:
            if month is None:
                for m in self.res_book.keys():
                    self.res_book[m].to_csv(
                        "results/number-{n}_month-{m}.csv".format(n=len(self.Users.keys()), m=m),
                        encoding='euc-kr'
                    )
            else:
                self.res_book[month].to_csv(
                    "results/number-{n}_month-{m}.csv".format(n=len(self.Users.keys()), m=month),
                    encoding='euc-kr'
                )
        elif typ == 2:
            concat_list = []
            for m in self.res_book.keys():
                if 'month' in self.res_book[m].columns:
                    self.res_book[m].drop(['month'], axis=1, inplace=True)
                self.res_book[m].insert(0, 'month', m)
                concat_list.append(self.res_book[m])

            df_all = pd.concat(concat_list, axis=0)
            df_all.to_excel(f"results/number-{len(self.Users.keys())}_month-all_{name}.xlsx")
