# -*- coding: utf-8 -*-
from datetime import datetime

import pandas as pd

from stockta.stockta import StockTreeAnalysis
from stockta.neostockbs import NoeStockBase

# ファイル読み込み
def read_k_db_data(fileName):
    # log
    _time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print "({}) INFO : start...".format(_time)

    # ファイル読み込み
    try:
        _sfb = pd.read_csv(fileName, encoding="shift_jis", index_col=0)
    except:
        _time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        print "({}) FATAL: ファイル読み込み失敗({})".format(_time, fileName)

    # 日本語のインデックスを英語化
    _sfb.columns = ["Open", "High", "Low", "Close", "Volume", "Trading Value"]
    _sfb.index.names = ["Day"]

    # indexを昇順にソート
    _sfb = _sfb.sortlevel()

    return _sfb

# main
if __name__ == "__main__":

    # k-dbのデータを読み込む
    fileName = "../data/stocks_7203-T_2.csv"
    sfb = read_k_db_data(fileName)

    # 株価データ加工
    stockbase = NoeStockBase(sfb)
    train_data, train_label = stockbase.processing()

    # 株価分析
    sta = StockTreeAnalysis()
    sta.grid_search(train_data, train_label)
    sta.cross_validation(train_data, train_label)
    #sta.fit(train_data, train_label)

    # グラフ出力
    sta.output_graph()

    # log
    _time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print "({}) INFO : ...end".format(_time)
