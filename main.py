# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

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

# 株式分析クラス
class StockTreeAnalysis(object):
    def __init__(self, data, label):
        self._data = data
        self._label = label

    def cross_validation(self):
        # 決定木分析
        clf = tree.DecisionTreeClassifier()
        scores = cross_validation.cross_val_score(clf, self._data, self._label, cv=5)

        # 結果出力
        print "正解率(平均)：{}".format(scores.mean())
        print "正解率(最小 / 最大)：{} / {}".format(scores.min(), scores.max())
        print "正解率(標準偏差)：{}".format(scores.std())
        print "正解率(全て)：{}".format(scores)

# main
if __name__ == "__main__":

    # k-dbのデータを読み込む
    fileName = "../data/stocks_7203-T.csv"
    sfb = ReadKDBData(fileName)

    # 単純移動平均
    sfb["SMA 5"] = ta.SMA(np.array(sfb["Close"]), timeperiod=5)
    sfb["SMA 25"] = ta.SMA(np.array(sfb["Close"]), timeperiod=25)
    sfb["SMA 75"] = ta.SMA(np.array(sfb["Close"]), timeperiod=75)

    # ボリンジャーバンド(2σ)
    sfb["Upper"], sfb["Middle"], sfb["Lower"] = ta.BBANDS(np.array(sfb["Close"]), timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # MACD
    sfb["MACD"], sfb["MACD Signal"], sfb["MACD Hist"] = ta.MACD(np.array(sfb["Close"]), fastperiod=12, slowperiod=26, signalperiod=9)

    # RSI
    sfb["RSI"] = ta.RSI(np.array(sfb["Close"]), timeperiod=14)

    # 加工データの保存
    sfb.to_csv("../result/ProcessingData.csv")

    # グラフ化
    '''
    fig, axes = plt.subplots(nrows=2, ncols=2)
    sfb[["Close", "SMA 5", "SMA 25", "SMA 75"]].plot(ax=axes[0,0]); axes[0,0].set_title("SMA")
    sfb[["Close", "Upper", "Middle", "Lower"]].plot(ax=axes[0,1]); axes[0,1].set_title("B Band")
    sfb[["MACD", "MACD Signal"]].plot(ax=axes[1,0]); axes[1,0].set_title("MACD")
    sfb["RSI"].plot(ax=axes[1,1]); axes[1,1].set_title("RSI")

    plt.tight_layout()
    plt.show()
    '''
    # ラベリング(負：0, 正：1)
    data = sfb["Close"].values
    label = np.zeros(len(data))
    for i in range(len(label)):
        if i == 0:
            label[0] = 0
        else:
            if data[i] > data[i-1]:
                label[i] = 1
            else:
                label[i] = 0

    sfb["label"] = label

    # 不要な行を削除
    sfb = sfb.ix[74:, :]

    # 加工データの分割
    train_data = sfb.ix[:, 0:16]
    train_label = sfb.ix[:, "label"]

    # 株価分析
    sta = StockTreeAnalysis(train_data, train_label)
    sta.cross_validation()
