# -*- coding: utf-8 -*-
import numpy as np

from sklearn.preprocessing import StandardScaler

# 指数作成クラス
class hslib(object):
    def __init__(self):
        self._stdsc = None

    def StandardScalerOfHSLib(self, data):
        self._stdsc = StandardScaler()
        _result = self._stdsc.fit_transform(data)
        return _result

    def Diff(self, data):
        _std_data = self.StandardScalerOfHSLib(data)
        _result = np.zeros(len(_std_data))
        for i in range(len(_std_data)):
            if i == 0:
                _result[i] = 0
            else:
                _result[i] = _std_data[i] - _std_data[i-1]
        return _result

    def Over(self, comparison, data):
        _result = np.zeros(len(data))
        for i in range(len(data)):
            if data[i] >= comparison[i]:
                _result[i] = True
            else:
                _result[i] = False
        return _result

    def Under(self, comparison, data):
        _result = np.zeros(len(data))
        for i in range(len(data)):
            if data[i] <= comparison[i]:
                _result[i] = True
            else:
                _result[i] = False
        return _result

    def GoldenCross(self, comparison, data):
        _result = np.zeros(len(data))
        for i in range(len(data)):
            if i == 0:
                _result[i] = False
            else:
                if data[i-1] < comparison[i-1] and data[i] >= comparison[i-1]:
                    _result[i] = True
                else:
                    _result[i] = False
        return _result
