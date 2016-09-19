# -*- coding: utf-8 -*-
from sklearn import tree
from sklearn import grid_search
from sklearn import cross_validation

# 株式分析クラス
class StockTreeAnalysis(object):
    def __init__(self, min_samples_leaf=2, max_depth=None):
        #self._data = data
        #self._label = label
        self._min_samples_leaf = min_samples_leaf
        self._max_depth = max_depth

    def grid_search(self, data, label):
        # ハイパーパラメータ
        _param = {
            "min_samples_leaf"  :[1, 5],
            "max_depth"         :[3, 10]
        }

        # 決定木分析
        _clf = grid_search.GridSearchCV(tree.DecisionTreeClassifier(), _param)
        _clf.fit(data, label)

        # 結果出力
        print "最適解：{}".format(_clf.best_estimator_)

    def cross_validation(self, data, label):
        # 決定木分析
        _clf = tree.DecisionTreeClassifier(min_samples_leaf=self._min_samples_leaf, max_depth = self._max_depth)
        _scores = cross_validation.cross_val_score(_clf, data, label, cv=4)

        # 結果出力
        print "正解率(平均)：{}".format(_scores.mean())
        print "正解率(最小 / 最大)：{} / {}".format(_scores.min(), _scores.max())
        print "正解率(標準偏差)：{}".format(_scores.std())
        print "正解率(全て)：{}".format(_scores)

        return _scores.mean()
