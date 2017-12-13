#coding:utf-8
import codecs  # 文件处理模块
import json  # JSON(JavaScript Object Notation) 是一种轻量级的数据交换格式，易于人阅读和编写
import math
import os  # 文件目录操作
import jieba  # 结巴分词分为三种模式：精确模式（默认）、全模式和搜索引擎模式
import nltk  # 自然语言处理用库
import DataSpider  # 爬虫


class Preprocessor:
    def __init__(self):
        self.features = []

# 获取数据
    def get_new_data(self):
        DataSpider.DataSpider().get_data()

    def process_data(self, pos_file='./corpus/pos.txt', neg_file='./corpus/neg.txt', feature_num=40):
        if not os.path.exists(pos_file) or not os.path.exists(neg_file):
            self.get_new_data()
        # process the data
        stopwords = self.loan_stopwords()  # 停止词表
        pos = self.loan_txt(pos_file)  # 加载好评
        neg = self.loan_txt(neg_file)  # 加载差评
        pos_seg_list = self.get_seg_list(pos, stopwords)  # 获取分词表
        neg_seg_list = self.get_seg_list(neg, stopwords)  # 获取分词表


        pos_freq_dist = self.get_freq_dist(pos_seg_list)  # 获取词频
        neg_freq_dist = self.get_freq_dist(neg_seg_list)  # 获取词频
        pos_words_set = self.get_words_set(pos_seg_list)  #
        neg_words_set = self.get_words_set(neg_seg_list)  #
        pos_words_num = self.compute_words_num(pos_seg_list)  # 词数
        neg_words_num = self.compute_words_num(neg_seg_list)  # 词数

        # print(pos_words_num, neg_words_num)
        total_seg_list = []  # 总集
        for each in pos_seg_list:
            total_seg_list.append(each)
        for each in neg_seg_list:
            total_seg_list.append(each)

        pos_words_tfidf = self.compute_TF_IDF(pos_words_set, pos_freq_dist, pos_words_num, total_seg_list)
        neg_words_tfidf = self.compute_TF_IDF(neg_words_set, neg_freq_dist, neg_words_num, total_seg_list)
        pos_words_tfidf = self.sort_by_value(pos_words_tfidf)
        neg_words_tfidf = self.sort_by_value(neg_words_tfidf)

        # pos_words_PMI = self.PMI(pos_words_set, pos_seg_list, len(pos_seg_list) / len(total_seg_list))  # 计算每一个词的PMI
        # neg_words_PMI = self.PMI(neg_words_set, neg_seg_list, len(neg_seg_list) / len(total_seg_list))  # 计算每一个词的PMI
        # pos_words_PMI = self.sort_by_value(pos_words_PMI)
        # neg_words_PMI = self.sort_by_value(neg_words_PMI)

        # pos_words_Chi_square = self.Chi_square(pos_words_set, pos_seg_list, total_seg_list)
        # neg_words_Chi_square = self.Chi_square(neg_words_set, neg_seg_list, total_seg_list)
        # pos_words_Chi_square = self.sort_by_value(pos_words_Chi_square)
        # neg_words_Chi_square = self.sort_by_value(neg_words_Chi_square)
        #self.pos = pos_words_Chi_square
        #self.neg = neg_words_Chi_square

        self.features = self.feature_list(pos_words_tfidf, feature_num, neg_words_tfidf, feature_num)
        self.save_feature_model(self.features)
        self.create_train_csv(self.features, pos_freq_dist, neg_freq_dist)
        return len(pos_seg_list) / len(total_seg_list)

    def save_feature_model(self, features, filename='./model/TD-TDF_40.json'):
        with codecs.open(filename, 'w', 'utf-8') as f:
            f.write(json.dumps(features, ensure_ascii=False))
            f.close()

    def load_feature_model(self, filename='./model/TD-TDF_40.json'):
        with codecs.open(filename, 'r', 'utf-8') as f:
            self.features = json.loads(f.read())

    ###################################
    def sentence2vec(self, sentence):
        if len(self.features) == 0:
            self.load_feature_model()
        seg_list = jieba.cut(sentence, False)
        freq_dist = nltk.FreqDist(seg_list)
        local_list = []
        for each in self.features:
            local_list.append(freq_dist[each])
        return local_list

    def loan_stopwords(self):
        stopwords = []
        with codecs.open('./corpus/stopwords.txt', 'r', 'utf-8') as f:
            for each in f.readlines():
                each = each.strip('\n')
                each = each.strip('\r')
                stopwords.append(each)
        return stopwords

    def loan_txt(self, filename):
        lists = []
        with codecs.open(filename, 'r', 'utf-8') as f:
            for each in f.readlines():
                if each != '':
                    lists.append(each.strip('\n'))
        return lists

    def get_seg_list(self, array, stopwords):
        seg_list = []
        for each in array:
            local_list = jieba.cut(each, False)
            final_list = []
            for word in local_list:
                if word not in stopwords and word != ' ':
                    final_list.append(word)
            seg_list.append(final_list)
        return seg_list

    def get_freq_dist(self, seg_list):
        freq_dist = []
        for each in seg_list:
            freq_dist.append(nltk.FreqDist(each))
        return freq_dist

    def get_words_set(self, seg_list):
        word_set = set()
        for each in seg_list:
            for word in each:
                word_set.add(word)
        return word_set

    def compute_words_num(self, seg_list):
        total = 0
        for each in seg_list:
            total += len(each)
        return total
    # pos_words_tfidf = self.compute_TF_IDF(pos_words_set, pos_freq_dist, pos_words_num, total_seg_list)
    # 好评词集合 好评词频 好评词总数 总词典
    def compute_TF_IDF(self, word_set, freq_dist, words_num, total_seg_list):
        word_dist = {}
        for word in word_set:
            tf = 0
            for each in freq_dist:  # 在好/差评中每一个word统计词频并除以总词数
                tf += each[word]
            tf /= words_num
            total_num = len(total_seg_list)  # 计算总词数
            exist_num = 0
            for each in total_seg_list:  # 在总评论中统计词频
                if word in each:
                    exist_num += 1
                    continue
            idf = math.log(total_num / exist_num)
            word_dist[word] = tf * idf
        return word_dist

    def sort_by_value(self, d):
        items = d.items()
        backitems = [[v[1], v[0]] for v in items]
        backitems.sort(reverse=True)
        return [backitems[i] for i in range(0, len(backitems))]

    def feature_list(self, pos_words_tfidf, pos_feature_num, neg_words_tfidf, neg_feature_num):
        features = []
        for each in pos_words_tfidf[:pos_feature_num]:
            features.append(each[1])
        for each in neg_words_tfidf[:neg_feature_num]:
            features.append(each[1])
        return features

    def create_train_csv(self, features, pos_freq_dist, neg_freq_dist):
        with codecs.open('./corpus/TD-TDF_40.csv', 'w', 'utf-8') as f:
            f.write(','.join(features) + ',sentiment\n')
            pos_words_vec = self.words2vec(features, pos_freq_dist)
            neg_words_vec = self.words2vec(features, neg_freq_dist)
            for each in pos_words_vec:
                f.write(','.join(each) + ',1\n')
            for each in neg_words_vec:
                f.write(','.join(each) + ',-1\n')
            f.close()

    ###################################
    def words2vec(self, features, freq_dist):
        word_vec = []
        for sentence in freq_dist:
            local_vec = []
            for each in features:
                local_vec.append(str(sentence[each]))
            word_vec.append(local_vec)
        return word_vec

        # pos_words_PMI = self.PMI(pos_words_set, pos_seg_list, len(pos_seg_list) / len(total_seg_list))  # 计算每一个词的PMI
        # neg_words_PMI = self.PMI(neg_words_set, neg_seg_list, len(neg_seg_list) / len(total_seg_list))  # 计算每一个词的PMI
        # pos_words_PMI = self.sort_by_value(pos_words_PMI)
        # neg_words_PMI = self.sort_by_value(neg_words_PMI)

    def PMI(self, words_set, seg_list, prob):
        PMI_words = {}
        for each in words_set:
            occur = 0
            for sent in seg_list:
                if each in sent:
                    occur += 1
            word_prob = occur / len(seg_list)
            PMI_words[each] = math.log(word_prob / prob)  # 有问题
        return PMI_words

    def Chi_square(self, words, seg_list, total_seg_list):
        total_num = len(total_seg_list)
        prob = len(seg_list) / len(total_seg_list)
        Chi_square_words = {}
        for word in words:
            occur_in_class = 0
            occur_in_all = 0
            for each in seg_list:
                if word in each:
                    occur_in_class += 1
            for each in total_seg_list:
                if word in each:
                    occur_in_all += 1
            prob_in_class = occur_in_class / len(seg_list)
            prob_in_all = occur_in_all / len(total_seg_list)
            Chi_square_words[word] = total_num * prob_in_all ** 2 * (prob_in_class - prob) ** 2 /\
                                    (prob_in_all * (1 - prob_in_all) * prob * (1 - prob))
        return Chi_square_words

if __name__ == '__main__':
    processor = Preprocessor()
    processor.process_data()
    print(processor.process_data())
    # print(processor.neg)

