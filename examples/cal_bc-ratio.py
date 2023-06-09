from lib.parameters import (
    ProbOptions,
    ReadInputData,
    SetTimes,
    UserInfo
)

from lib.parameters import (
    make_generation_pattern
)

from lib.auction_oper_model import (
    market_clearing
)


if __name__ == '__main__':
    options = ProbOptions()
    IFN = ReadInputData(options)
    Gen_prob_pattern = make_generation_pattern(IFN)

    options.month = 1
    ST = SetTimes(options, IFN)
    UI = UserInfo(options, IFN, ST, Gen_prob_pattern)

    auction_book, users_book = market_clearing(
        options,
        IFN,
        ST,
        UI
    )




    # 이후, prosumer, non-prosumer, utility 별 B/C ratio 산정하기
    # b/c ratio 산정에 대해서는 아래와 같음
    # P2P 시장을 하지 않았을 때의 비용 대비 P2P 시장을 open 함으로써 얻는 편익에 대한 ratio를 의미함
    # utility의 경우에는 다음과 같음
    # p2p를 하지 않았을 때 --> (전기요금 - p2p VCG로 인한 gap 손실) / (전기요금 - 상계거래로 인한 누적손실)

    # CASE Study 진행
    # 총 고객 중 프로슈머가 몇명일 때? 10명 중 5명 vs 10명 중 8명
    # 전기요금 수준이 높은 village vs 전기요금이 수준이 낮은 village
