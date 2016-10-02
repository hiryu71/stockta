# -*- coding: utf-8 -*-
import numpy as np
import talib as ta
import pandas as pd
import matplotlib.pyplot as plt

from stockbs import StockBase
from hslib import hslib

# 株式データ加工クラス
class NoeStockBase(StockBase):
    def __init__(self, data):
        self._data = data
        self._data2 = None

    def processing(self):
        # 単純移動平均
        self._data["EMA 5"] = ta.EMA(np.array(self._data["Close"]), timeperiod=5)
        self._data["EMA 25"] = ta.EMA(np.array(self._data["Close"]), timeperiod=25)
        self._data["EMA 75"] = ta.EMA(np.array(self._data["Close"]), timeperiod=75)

        # ボリンジャーバンド(2σ)
        self._data["Upper"], self._data["Middle"], self._data["Lower"] = ta.BBANDS(np.array(self._data["Close"]), timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        # MACD
        self._data["MACD"], self._data["MACD Signal"], _tmp = ta.MACD(np.array(self._data["Close"]), fastperiod=12, slowperiod=26, signalperiod=9)

        # RSI
        self._data["RSI 7"] = ta.RSI(np.array(self._data["Close"]), timeperiod=7)
        self._data["RSI 14"] = ta.RSI(np.array(self._data["Close"]), timeperiod=14)
        self._data["RSI 28"] = ta.RSI(np.array(self._data["Close"]), timeperiod=28)

        # 欠損値がある行を削除
        self._data = self._data.dropna()

        # hslib
        self._data2 = self._data.copy()
        hs = hslib()
        self._data2["Diff EMA 5"] = hs.Diff(self._data2["EMA 5"])
        self._data2["Diff EMA 25"] = hs.Diff(self._data2["EMA 25"])
        self._data2["Diff EMA 75"] = hs.Diff(self._data2["EMA 75"])

        self._data2["RSI 7 * Diff EMA 5"] = self._data["RSI 7"] * self._data2["Diff EMA 5"]
        self._data2["RSI 7 * Diff EMA 25"] = self._data["RSI 7"] * self._data2["Diff EMA 25"]
        self._data2["RSI 7 * Diff EMA 75"] = self._data["RSI 7"] * self._data2["Diff EMA 75"]

        self._data2["Over Upper at Close"] = hs.Over(self._data["Upper"], self._data["Close"])
        self._data2["Under Lower at Close"] = hs.Under(self._data["Lower"], self._data["Close"])

        self._data2["Golden Cross of EMA 5"] = hs.GoldenCross(self._data["EMA 5"], self._data["Close"])


        # ラベリング(負：0, 正：1)
        self._labeling()

        # 不要な行を削除
        #_pick_columns = ["Close", "EMA 5", "EMA 25", "EMA 75", "RSI 7", "RSI 14", "RSI 28", "label"]
        #self._pick_columns(_pick_columns)
        _drop_columns = ["High", "Volume"]
        self._drop_columns(_drop_columns)

        # 加工データの保存
        self._data.to_csv("../result/ProcessingData1.csv")
        self._data2.to_csv("../result/ProcessingData2.csv")

        # グラフ
        #self.plot()
        #self.plot2()

        # 加工データの分割
        _tmp, _column_num = self._data2.shape
        _train_data = self._data2.ix[:, 0:_column_num-1]
        _train_label = self._data2.ix[:, "label"]

        return _train_data, _train_label

    def _labeling(self):
        # ラベリング(Down：0, Up：1)
        _data = self._data2["Close"].values
        _label = np.zeros(len(_data))
        for i in range(len(_label)):
            if i == len(_label)-1:
                _label[i] = StockBase.LABEL_DOWN
            else:
                if _data[i+1] > _data[i]:
                    _label[i] = StockBase.LABEL_UP
                else:
                    _label[i] = StockBase.LABEL_DOWN

        self._data2["label"] = _label

    def _pick_columns(self, columns):
        self._data2 = self._data2.ix[:, columns]
        print "---------------------------"
        print "分析項目：{}".format(self._data2.columns.values)
        #print "{}".format(self._data2.head(4))


    def _drop_columns(self, columns):
        self._data2 = self._data2.drop(columns, axis=1)
        print "---------------------------"
        print "分析項目：{}".format(self._data2.columns.values)
        print "分析対象外項目：{}".format(columns)
        #print "{}".format(self._data2.head(4))

    def plot2(self):
        _fig, _axes = plt.subplots(nrows=2, ncols=2)
        self._data[["Close", "EMA 5", "EMA 25", "EMA 75"]].plot(ax=_axes[0,0]); _axes[0,0].set_title("EMA")
        self._data[["Close", "Upper", "Middle", "Lower"]].plot(ax=_axes[0,1]); _axes[0,1].set_title("B Band")
        self._data2[["Diff EMA 5", "Diff EMA 25", "Diff EMA 75"]].plot(ax=_axes[1,0]); _axes[1,0].set_title("ROC EMA")
        self._data2[["RSI 7", "RSI 14", "RSI 28"]].plot(ax=_axes[1,1]); _axes[1,1].set_title("RSI")

        plt.tight_layout()
        plt.show()
