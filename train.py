#coding:utf-8
import codecs
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # 版本异常
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import column_or_1d
from sklearn.utils import shuffle
import Preprocessor
import matplotlib as mpl

# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'Microsoft YaHei'
sns.set_style("whitegrid")


class SentimentClassifier:
    def __init__(self, filename='./corpus/TD-TDF_30.csv'):
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            self.data = shuffle(data)
            #print(data)
            # 删除sentiment列
            X_data = pd.DataFrame(data.drop('sentiment', axis=1))
            Y_data = column_or_1d(data[:]['sentiment'], warn=True)
            self.ydata = Y_data
            # \ 续行
            self.X_train, self.X_val,\
            self.y_train, self.y_val = train_test_split(X_data, Y_data, test_size=0.3, random_state=1)
            # train_data：所要划分的样本特征集
            # train_target：所要划分的样本结果
            # test_size：样本占比，如果是整数的话就是样本的数量
            # random_state：是随机数的种子。
            # 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
            # 比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
            # 随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
            # 种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
            self.model = None
            self.load_model()
            self.preprocessor = Preprocessor.Preprocessor()
        else:
            print('No Source!')
            self.preprocessor.process_data()

    def load_model(self, filename='./model/TD-TDF_30_model.pickle'):
        if os.path.exists(filename):
            with codecs.open(filename, 'rb') as f:
                f = open(filename, 'rb')
                self.model = pickle.load(f)
        else:
            self.train()

    def save_model(self, filename='./model/TD-TDF_30_model.pickle'):
        with codecs.open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def train(self):
            self.model = LogisticRegression(random_state=7)  # 预定义
            #self.model = RandomForestClassifier(random_state=7)
            #self.model = GaussianNB()  # 预定义
            #self.model = LinearSVC(random_state=7)
            self.model.fit(self.X_train, self.y_train)
            self.save_model()
            self.model.score(self.X_val, self.y_val)
            print('Accuracy: ' + str(round(self.model.score(self.X_val, self.y_val), 2)))

    def predict(self, sentence):
        vec = self.preprocessor.sentence2vec(sentence)
        return self.model.predict(vec)

    def predict_test_set(self, sentences, pos_file='./test/pos_test.txt', neg_file='./test/neg_test.txt'):
        pos_set = []
        neg_set = []
        for each in sentences:
            score = self.predict(each)
            if score == 1:
                pos_set.append(each)
                print('pos')
            elif score == -1:
                neg_set.append(each)
                print('neg')
        with codecs.open(pos_file, 'w', 'utf-8') as f:
            for each in pos_set:
                f.write(each + '\n')
            f.close()
        with codecs.open(neg_file, 'w', 'utf-8') as f:
            for each in neg_set:
                f.write(each + '\n')
            f.close()

    def show_heat_map(self):
            pd.set_option('precision', 2)
            plt.figure(figsize=(20, 6))
            sns.heatmap(self.data.corr(), square=True)
            plt.xticks(rotation=90, fontproperties='Microsoft YaHei')
            plt.yticks(rotation=360, fontproperties='Microsoft YaHei')
            plt.suptitle("Correlation Heatmap", fontsize=20)
            plt.show()

    def show_heat_map_to(self, target='sentiment'):
            correlations = self.data.corr()[target].sort_values(ascending=False)
            plt.figure(figsize=(40, 6))
            correlations.drop(target).plot.bar()
            pd.set_option('precision', 2)
            plt.xticks(rotation=90, fontsize=14, fontproperties='Microsoft YaHei')
            plt.yticks(rotation=360, fontproperties='Microsoft YaHei')
            plt.suptitle('The Heatmap of Correlation With ' + target)
            plt.show()

    # ################重点##################
    def plot_learning_curve(self):
        # Plot the learning curve
        plt.figure(figsize=(9, 6))
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X=self.X_train, y=self.y_train,
            cv=3, scoring='neg_mean_squared_error')
        #, scoring='neg_mean_squared_error'
        self.plot_learning_curve_helper(train_sizes, train_scores, test_scores, 'Learning Curve')
        plt.show()

    def plot_learning_curve_helper(self, train_sizes, train_scores, test_scores, title, alpha=0.1):
        train_scores = -train_scores
        test_scores = -test_scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean + train_std,
                         train_mean - train_std, color='blue', alpha=alpha)
        plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
        plt.title(title)
        plt.xlabel('Number of training points')
        plt.ylabel(r'Accuracy')
        plt.grid(ls='--')
        plt.legend(loc='best')
        plt.show()

    # def feature_reduction(self, X_train, y_train, X_val):
    #     thresh = 5 * 10 ** (-3)
    #     # model = XGBRegressor()
    #     model.fit(X_train, y_train)
    #     selection = SelectFromModel(model, threshold=thresh, prefit=True)
    #     select_X_train = selection.transform(X_train)
    #     select_X_val = selection.transform(X_val)
    #     return select_X_train, select_X_val

    def choose_best_model(self):
        seed = 7
        pipelines = []
        pipelines.append(
            ('SVC',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ("SVC", SVC(random_state=seed))
             ])
             )
        )
        pipelines.append(
            ('AdaBoostClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('AdaBoostClassifier', AdaBoostClassifier(random_state=seed))
             ]))
        )
        pipelines.append(
            ('RandomForestClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('RandomForestClassifier', RandomForestClassifier(random_state=seed))
             ]))
        )
        #pipelines.append(
         #   ('RandomForestClassifier',
          #   Pipeline([
           #      ('Scaler', StandardScaler()),
            #     ('RandomForestClassifier', RandomForestClassifier(random_state=seed))
             #]))
        #)
        pipelines.append(
            ('LinearSVC',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('LinearSVC', LinearSVC(random_state=seed))
             ]))
        )
        pipelines.append(
            ('KNeighborsClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('KNeighborsClassifier', KNeighborsClassifier())
             ]))
        )

        pipelines.append(
            ('GaussianNB',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('GaussianNB', GaussianNB())
             ]))
        )

        #pipelines.append(
        #    ('Perceptron',
        #     Pipeline([
        #         ('Scaler', StandardScaler()),
        #         ('Perceptron', Perceptron(random_state=seed))
        #     ]))
        #)
        #pipelines.append(
        #    ('SGDClassifier',
        #     Pipeline([
        #         ('Scaler', StandardScaler()),
        #         ('SGDClassifier', SGDClassifier(random_state=seed))
        #     ]))
        #)
        pipelines.append(
            ('DecisionTreeClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=seed))
             ]))
        )
        pipelines.append(
            ('LogisticRegression',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('LogisticRegression', LogisticRegression(random_state=seed))
             ]))
        )
        scoring = 'accuracy'
        n_folds = 10
        results, names = [], []
        for name, model in pipelines:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            cv_results = cross_val_score(model, self.X_train, self.y_train, cv=kfold,
                                         scoring=scoring, n_jobs=-1)
            names.append(name)
            results.append(cv_results)
            msg = "%s: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)


if __name__ == '__main__':
    #DataSpider.Preprocessor().get_new_data()
    classifier = SentimentClassifier()
    classifier.train()
    #print(classifier.ydata)
    #print(classifier.data)
    #print(classifier.X_train)
    #print(classifier.X_val)
    #print(classifier.y_train)
    #print(classifier.y_val)
    #print(classifier.data)
    classifier.plot_learning_curve()
    #classifier.show_heat_map()
    #classifier.show_heat_map_to()
    #classifier.choose_best_model()
    #print(classifier.predict(array('我喜欢这个手机').reshape(1, -1)))
    # test_set = []
    # with codecs.open('./test/test.txt', 'r', 'utf-8') as f:
    #     for each in f.readlines():
    #         test_set.append(each)
    # classifier.predict_test_set(test_set)
