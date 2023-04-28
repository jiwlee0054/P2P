import lib.parameters as para
import lib.auction_oper_model as aom

if __name__ == '__main__':
    options = para.ProbOptions()
    IFN = para.ReadInputData(options)
    Gen_prob_pattern = para.make_generation_pattern(IFN)
    ST = para.SetTimes(options)
    UI = para.UserInfo(options, IFN, ST, Gen_prob_pattern)

    AOM = aom.Auction(options, IFN, ST, UI)

    print("working")