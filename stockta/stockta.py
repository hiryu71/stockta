# -*- coding: utf-8 -*-
from sklearn import tree
from sklearn import cross_validation

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
