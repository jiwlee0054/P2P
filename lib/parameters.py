import pandas as pd
import numpy as np
import json
import random

from datetime import datetime, timedelta
from dateutil import relativedelta


class ProbOptions:
    def __init__(self):
        self.Number_Of_Users = 100
        self.Number_Of_Consumers = 50
        self.Number_Of_Prosumers = 50

        self.year = 2021
        self.month = 1
        self.season = self.set_season()

        self.demand_noise_dist_deviation_ratio = 0.3
        self.trading_time_option = 'Time-of-Use'
        self.auction_mechanism = 'VCG'

        """
        입찰 전략 선택
        1. Randomly
        2. Update margin
        """
        self.bidding_strategy = 'Update margin'

        self.loc_data = f"../data"

    def set_season(self):
        if self.month == 7 or self.month == 8:
            return '하계'
        else:
            return '기타'

class ReadInputData:
    def __init__(self, options: ProbOptions):
        self.demand_factor = pd.read_excel(f"{options.loc_data}/demand.xlsx", sheet_name='주택용', index_col=0)
        self.consumption_mean, self.consumption_std = pd.read_excel(f"{options.loc_data}/electricity_consumption_normal_parameter.xlsx",
                                                                    index_col=0).loc[options.month, :].values
        with open(f"{options.loc_data}/excess_electricity_sample_set.json", "r") as f:
            self.excess_electricity_sample_set = json.load(f)
        self.excess_electricity_sample_set = self.excess_electricity_sample_set[f"{options.month}월"]

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

        # pv_cap_sample_set.append([(0 + 1) / 2] * int(pv_cap_historic.loc['1kW 이하', '2020'] / ((1 + 3) / 2)))
        # pv_cap_sample_set.append([(1 + 3) / 2] * int(pv_cap_historic.loc['1~3kW 이하', '2020'] / ((1 + 3) / 2)))
        # pv_cap_sample_set.append([(3 + 10) / 2] * int(pv_cap_historic.loc['3~10kW 이하', '2020'] / ((3 + 10) / 2)))
        # pv_cap_sample_set.append([(10 + 50) / 2] * int(pv_cap_historic.loc['10~50kW 이하', '2020'] / ((10 + 50) / 2)))
        # pv_cap_sample_set.append([(50 + 100) / 2] * int(pv_cap_historic.loc['50~100kW 이하', '2020'] / ((50 + 100) / 2)))
        # pv_cap_sample_set.append([(100 + 500) / 2] * int(pv_cap_historic.loc['100~500kW 이하', '2020'] / ((100 + 500) / 2)))
        # pv_cap_sample_set.append([(500 + 1000) / 2] * int(pv_cap_historic.loc['500~1,000kW 이하', '2020'] / ((500 + 1000) / 2)))
        self.pv_cap_sample_set = sum(pv_cap_sample_set, [])
        # self.pv_cap_mean, self.pv_cap_std = np.array(pv_cap_sample_set).mean(), np.array(pv_cap_sample_set).std()

        self.tariff_table = pd.read_excel(f"{options.loc_data}/tariff_table.xlsx", sheet_name='주택용(22.10.01)', index_col=0)

        smp_data = pd.read_excel(f"{options.loc_data}/smp_land_2019.xlsx", sheet_name='smp_land_2019', index_col=0, header=1)
        smp_index = [f"{k.replace('2019', '')[0:2]}-{k.replace('2019', '')[2:]}" for k in smp_data.index.astype('str')]
        self.smp_data = pd.DataFrame(smp_data.values, index=smp_index, columns=smp_data.columns)

        self.PF_mean, self.PF_sigma = 0.2, 0.15

def make_generation_pattern(IFN):
    generation_pat = IFN.pv_gen_sample_set

    ind_ = range(1, 366, 1)
    col_ = range(1, 25, 1)

    Gen_prob_pattern = pd.DataFrame(index=ind_, columns=col_)
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
            Gen_prob_pattern.loc[i, j] = val_

    return Gen_prob_pattern


class SetTimes:
    def __init__(self, options: ProbOptions):
        this_month = datetime(year=options.year, month=options.month, day=1).date()
        next_month = this_month + relativedelta.relativedelta(months=1)
        self.first_day = this_month
        self.last_day = next_month - timedelta(days=1)

        self.set_trading_time(options)

    def set_trading_time(self, options):
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


class UserInfo:
    def __init__(self, options: ProbOptions, IFN:ReadInputData, ST: SetTimes, Gen_prob_pattern):
        self.items = ['Gen_bin',
                      'Electricity_Consumption',
                      'Generation',
                      'Block_step_old',
                      'Total_amount_of_consumption',
                      'Demand_charge',
                      'Energy_charge',
                      'Installed_Cap'
                      ]

        self.first_day = ST.first_day
        self.last_day = ST.last_day

        # Users SET 만들기
        self.make_users_set(options)
        self.put_consumption_to_Users(options, IFN)
        self.put_generation_to_Users(Gen_prob_pattern, IFN)
        self.put_TariffInfo_to_Users(options, IFN)

    def make_users_set(self, options):
        Users = dict()
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

        Users_sorted = dict()
        for u in sorted(Users.items(), key=lambda item: int(item[0][1:])):
            Users_sorted[u[0]] = u[1]

        self.Users = Users_sorted

    def put_consumption_to_Users(self, options, IFN):
        for u in self.Users.keys():
            Demand_date_factor = self.make_random_demand_factor(options, IFN)
            consumption_pattern, electricity_amount = self.make_electricity_consumption_pattern_by_user(IFN, Demand_date_factor)
            self.Users[u]['Electricity_Consumption'] = consumption_pattern
            self.Users[u]['Total_amount_of_consumption'] = electricity_amount

    def put_generation_to_Users(self, Gen_prob_pattern, IFN):
        gen_prob_pattern_dict = self.make_gen_pattern_dict(Gen_prob_pattern)
        for u in self.Users.keys():
            if self.Users[u]['Gen_bin'] == 1:
                while True:
                    # cap_rand = np.round(np.random.normal(loc=IFN.pv_cap_mean, scale=IFN.pv_cap_std), 1)
                    cap_rand = np.round(random.choice(IFN.pv_cap_sample_set), 2)
                    if cap_rand >= 0:
                        break
                    else:
                        pass
                self.Users[u]['Generation'] = dict(zip(gen_prob_pattern_dict.keys(), map(lambda x:x[1] * cap_rand,
                                                                                         gen_prob_pattern_dict.items())))
                self.Users[u]['Installed_Cap'] = cap_rand
            else:
                self.Users[u]['Installed_Cap'] = 0

    def put_TariffInfo_to_Users(self, options, IFN):
        for u in self.Users.keys():
            contract_voltage = max(self.Users[u]['Electricity_Consumption'].values())
            if contract_voltage <= 3:
                vol_level = '저압'
            else:
                vol_level = '고압'
            amount = self.Users[u]['Total_amount_of_consumption']
            block = IFN.tariff_table[IFN.tariff_table['비고'] == options.season][amount < IFN.tariff_table[IFN.tariff_table['비고'] == options.season].loc[:, "max"]].iloc[0, :]
            step = block.name
            demand_charge = block[f"{vol_level}, 기본"]

            block_lower = IFN.tariff_table[IFN.tariff_table['비고'] == options.season][amount >= IFN.tariff_table[IFN.tariff_table['비고'] == options.season].loc[:, "max"]]
            energy_charge = 0
            max_val = 0
            for i in block_lower.index:
                energy_charge += (block_lower.loc[i, 'max'] - max_val) * block_lower.loc[i, f"{vol_level}, 전력량"]
                max_val = block_lower.loc[i, 'max']

            energy_charge += (amount - max_val) * block[f"{vol_level}, 전력량"]

            self.Users[u]['Block_step_old'] = int(step)
            self.Users[u]['Demand_charge'] = int(demand_charge)
            self.Users[u]['Energy_charge'] = float(energy_charge)

    def make_gen_pattern_dict(self, Gen_prob_pattern):
        gen_prob_pattern_dict = dict()
        for d in range(self.first_day.day, self.last_day.day + 1):
            for h in range(1, 25):
                gen_prob_pattern_dict[d, h] = Gen_prob_pattern.loc[d, h]

        return gen_prob_pattern_dict

    def make_random_demand_factor(self, options, IFN):
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

    def make_electricity_consumption_pattern_by_user(self, IFN, Demand_date_factor):
        while True:
            electricity_amount = np.random.normal(IFN.consumption_mean, IFN.consumption_std)
            if electricity_amount > 0:
                break
            else:
                pass

        consumption_pattern = dict()
        factor_sum = sum(Demand_date_factor.values())
        for k, v in Demand_date_factor.items():
            # d = int(k.split(',')[0].split('(')[1])
            # h = int(k.split(',')[1].split(')')[0])
            consumption_pattern[k] = Demand_date_factor[k] * (electricity_amount / factor_sum)

        return consumption_pattern, electricity_amount
