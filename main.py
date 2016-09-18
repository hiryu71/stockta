# -*- coding: utf-8 -*-
import pandas as pd

from stockta.stockta import StockTreeAnalysis
from stockta.stockbs import StockBase

# ファイル読み込み
def read_k_db_data(fileName):

    # ファイル読み込み
    try:
        _sfb = pd.read_csv(fileName, encoding="shift_jis", index_col=0)
    except:
        print"Error: ファイル読み込み失敗({})".format(fileName)

    # 日本語のインデックスを英語化
    _sfb.columns = ["Open", "High", "Low", "Close", "Volume", "Trading Value"]
    _sfb.index.names = ["Day"]

    # indexを昇順にソート
    _sfb = _sfb.sortlevel()

    return _sfb

# main
if __name__ == "__main__":

    # k-dbのデータを読み込む
    fileName = "../data/stocks_7203-T.csv"
    sfb = read_k_db_data(fileName)

    # 株価データ加工
    stockbase = StockBase(sfb)
    train_data, train_label = stockbase.processing()

    # 株価分析
    sta = StockTreeAnalysis(train_data, train_label)
    sta.cross_validation()
