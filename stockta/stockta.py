# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

# 株式データ加工クラス
class StockBase(object):
    def __init__(self, data):
        self._data = data

    def processing(self):
        # 単純移動平均
        self._data["SMA 5"] = ta.SMA(np.array(self._data["Close"]), timeperiod=5)
        self._data["SMA 25"] = ta.SMA(np.array(self._data["Close"]), timeperiod=25)
        self._data["SMA 75"] = ta.SMA(np.array(self._data["Close"]), timeperiod=75)

        # ボリンジャーバンド(2σ)
        self._data["Upper"], self._data["Middle"], self._data["Lower"] = ta.BBANDS(np.array(self._data["Close"]), timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        # MACD
        self._data["MACD"], self._data["MACD Signal"], self._data["MACD Hist"] = ta.MACD(np.array(self._data["Close"]), fastperiod=12, slowperiod=26, signalperiod=9)

        # RSI
        self._data["RSI"] = ta.RSI(np.array(self._data["Close"]), timeperiod=14)

        # 加工データの保存
        self._data.to_csv("../result/ProcessingData.csv")

        # グラフ
        #self.plot()

        # ラベリング(負：0, 正：1)
        self._labeling()

        # 不要な行を削除
        self._data = self._data.ix[74:, :]

        # 加工データの分割
        _train_data = self._data.ix[:, 0:16]
        _train_label = self._data.ix[:, "label"]

        return _train_data, _train_label

    def _labeling(self):
        # ラベリング(負：0, 正：1)
        _data = self._data["Close"].values
        _label = np.zeros(len(_data))
        for i in range(len(_label)):
            if i == 0:
                _label[0] = 0
            else:
                if _data[i] > _data[i-1]:
                    _label[i] = 1
                else:
                    _label[i] = 0

        self._data["label"] = _label


    def plot(self):
        _fig, _axes = plt.subplots(nrows=2, ncols=2)
        self._data[["Close", "SMA 5", "SMA 25", "SMA 75"]].plot(ax=_axes[0,0]); _axes[0,0].set_title("SMA")
        self._data[["Close", "Upper", "Middle", "Lower"]].plot(ax=_axes[0,1]); _axes[0,1].set_title("B Band")
        self._data[["MACD", "MACD Signal"]].plot(ax=_axes[1,0]); _axes[1,0].set_title("MACD")
        self._data["RSI"].plot(ax=_axes[1,1]); _axes[1,1].set_title("RSI")

        plt.tight_layout()
        plt.show()


# 株式分析クラス
class StockTreeAnalysis(object):
    def __init__(self, data, label):
        self._data = data
        self._label = label

    def cross_validation(self):
        # 決定木分析
        _clf = tree.DecisionTreeClassifier()
        _scores = cross_validation.cross_val_score(_clf, self._data, self._label, cv=4)

        # 結果出力
        print "正解率(平均)：{}".format(_scores.mean())
        print "正解率(最小 / 最大)：{} / {}".format(_scores.min(), _scores.max())
        print "正解率(標準偏差)：{}".format(_scores.std())
        print "正解率(全て)：{}".format(_scores)

        return _scores.mean()
