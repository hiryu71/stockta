# -*- coding: utf-8 -*-
import numpy as np
import talib as ta
import matplotlib.pyplot as plt

# 株式データ加工クラス
class StockBase(object):
    LABEL_DOWN = 0
    LABEL_UP = 1

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
        # ラベリング(Down：0, Up：1)
        _data = self._data["Close"].values
        _label = np.zeros(len(_data))
        for i in range(len(_label)):
            if i == len(_label)-1:
                _label[i] = StockBase.LABEL_DOWN
            else:
                if _data[i+1] > _data[i]:
                    _label[i] = StockBase.LABEL_UP
                else:
                    _label[i] = StockBase.LABEL_DOWN

        self._data["label"] = _label


    def plot(self):
        _fig, _axes = plt.subplots(nrows=2, ncols=2)
        self._data[["Close", "SMA 5", "SMA 25", "SMA 75"]].plot(ax=_axes[0,0]); _axes[0,0].set_title("SMA")
        self._data[["Close", "Upper", "Middle", "Lower"]].plot(ax=_axes[0,1]); _axes[0,1].set_title("B Band")
        self._data[["MACD", "MACD Signal"]].plot(ax=_axes[1,0]); _axes[1,0].set_title("MACD")
        self._data["RSI"].plot(ax=_axes[1,1]); _axes[1,1].set_title("RSI")

        plt.tight_layout()
        plt.show()
