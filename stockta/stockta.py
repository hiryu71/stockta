# -*- coding: utf-8 -*-
from datetime import datetime

from sklearn import tree
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.externals.six import StringIO
import pydot
import subprocess

# 株式分析クラス
class StockTreeAnalysis(object):
    def __init__(self, min_samples_leaf=2, max_depth=None):
        self._clf = None
        self._data = {}
        self._label = {}
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf

    def grid_search(self, data, label):
        self._data = data
        self._label = label

        # パラメータ割り振り
        _params = {
            "min_samples_leaf"  :[1, 5],
            "max_depth"         :[3, 10]
        }

        # 決定木分析
        self._clf = grid_search.GridSearchCV(tree.DecisionTreeClassifier(), _params)
        self._clf.fit(self._data, self._label)

        # パラメータ設定
        self._max_depth, self._min_samples_leaf = self._clf.best_params_.values()

        # 結果出力
        print "---------------------------"
        print "最適解(成績)：{}".format(self._clf.best_score_)
        print "最適解(パラメータ)：{}".format(self._clf.best_params_)


    def cross_validation(self, data, label):
        self._data = data
        self._label = label

        # 決定木分析
        self._clf = tree.DecisionTreeClassifier(min_samples_leaf=self._min_samples_leaf, max_depth = self._max_depth)
        _scores = cross_validation.cross_val_score(self._clf, self._data, self._label, cv=4)

        # 結果出力
        print "---------------------------"
        print "正解率(平均)：{}".format(_scores.mean())
        print "正解率(最小 / 最大)：{} / {}".format(_scores.min(), _scores.max())
        print "正解率(標準偏差)：{}".format(_scores.std())
        print "正解率(全て)：{}".format(_scores)

        return _scores.mean()

    def fit(self, data, label):
        self._data = data
        self._label = label
        self._clf = tree.DecisionTreeClassifier(min_samples_leaf=self._min_samples_leaf, max_depth = self._max_depth)
        self._clf.fit(self._data, self._label)

    def output_graph(self):
        try:
            # 学習結果を可視化
            _dot_data = StringIO()
            tree.export_graphviz(self._clf, out_file=_dot_data, feature_names=self._data.columns, class_names=["Down","Up"], filled=True, rounded=True)
            _graph = pydot.graph_from_dot_data(_dot_data.getvalue())
            _graph.write_pdf("../result/tree.pdf")
            cmd = "open ../result/tree.pdf"
            subprocess.call(cmd, shell=True)
        except:
            _time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            print "({}) WARN : グラフ出力失敗".format(_time)

#, feature_names=iris.feature_names, class_names=iris.target_names
