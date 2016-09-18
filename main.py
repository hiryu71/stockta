# -*- coding: utf-8 -*-
import pandas as pd

from stockta.stockta import StockTreeAnalysis, StockBase

# ファイル読み込み
def ReadKDBData(fileName):

    # ファイル読み込み
    sfb = pd.read_csv(fileName, encoding="shift_jis", index_col=0)

    # 日本語のインデックスを英語化
    sfb.columns = ["Open", "High", "Low", "Close", "Volume", "Trading Value"]
    sfb.index.names = ["Day"]

    # indexを昇順にソート
    sfb = sfb.sortlevel()

    return sfb

# main
if __name__ == "__main__":

    # k-dbのデータを読み込む
    fileName = "../data/stocks_7203-T.csv"
    sfb = ReadKDBData(fileName)

    # 株価データ加工
    stockbase = StockBase(sfb)
    train_data, train_label = stockbase.processing()

    # 株価分析
    sta = StockTreeAnalysis(train_data, train_label)
    sta.cross_validation()
