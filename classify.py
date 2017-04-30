#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/4/23 17:57
@annotation = ''
"""
import random

import nltk
from nltk.classify import apply_features
from nltk.corpus import names, brown

labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])

random.shuffle(labeled_names)


def gender_features(word):
    return {'last_letter': word[-1]}


def classify():
    # 获取特征值
    # featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

    # 减少内存 返回一个行为像一个链表而不会在内存存储所有特征集的对象
    train_set = apply_features(gender_features, labeled_names[500:])
    test_set = apply_features(gender_features, labeled_names[:500])
    # train_set, test_set = featuresets[500:], featuresets[:500]

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(classifier.classify(gender_features('Neo')))
    print(nltk.classify.accuracy(classifier, test_set))
    print(classifier.show_most_informative_features(5))


"""
特征集不能太多 算法高度依赖训练集(过拟合)
完善特征集
    完善特征集的一个非常有成效的方法是错误分析
    训练集用于训练模型
    开发测试集用于进行错误分析
    测试集用于系统的最终评估
"""


def gender_features2(name):
    """特征集不能太多"""
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features


def better_classify():
    def gender_features(word):
        return {'suffix1': word[-1:],
                'suffix2': word[-2:]}

    train_names = labeled_names[1500:]
    devtest_names = labeled_names[500:1500]
    test_names = labeled_names[:500]

    train_set = [(gender_features(n), gender) for (n, gender) in train_names]
    devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
    test_set = [(gender_features(n), gender) for (n, gender) in test_names]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, devtest_set))

    """
    错误分析
        通过对比结果和实际情况 fix 特征集函数 提高accuracy
    """
    errors = []
    for (name, tag) in devtest_names:
        guess = classifier.classify(gender_features(name))
        if guess != tag:
            errors.append((tag, guess, name))
    for (tag, guess, name) in sorted(errors):
        print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))


def doc_classify():
    from nltk.corpus import movie_reviews
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    # 前 2000 个最频繁词的链表
    all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    word_features = list(all_words)[:2000]

    def document_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in document_words)
        return features

    document_features(movie_reviews.words('pos/cv957_8737.txt'))

    featuresets = [(document_features(d), c) for (d, c) in documents]
    print(featuresets)
    train_set, test_set = featuresets[100:], featuresets[:100]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    print(classifier.show_most_informative_features(5))


def explot_context():
    """探索上下文语境"""

    def pos_features(sentence, i):
        features = {"suffix(1)": sentence[i][-1:],
                    "suffix(2)": sentence[i][-2:],
                    "suffix(3)": sentence[i][-3:]}
        if i == 0:
            features["prev-word"] = "<START>"
        else:
            features["prev-word"] = sentence[i - 1]
        return features

    # pos_features(brown.sents()[0], 8)
    tagged_sents = brown.tagged_sents(categories='news')
    featuresets = []
    for tagged_sent in tagged_sents:
        untagged_sent = nltk.tag.untag(tagged_sent)
        for i, (word, tag) in enumerate(tagged_sent):
            featuresets.append((pos_features(untagged_sent, i), tag))

    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    nltk.classify.accuracy(classifier, test_set)


"""
评估:
    准备测试集
    测试集准确率

    真阳性True positives 是相关项目中我们正确识别为相关的。
    真阴性True negatives 是不相关项目中我们正确识别为不相关的。
    假阳性False positives (或 I 型错误)是不相关项目中我们错误识别为相关的。
    假阴性False negatives (或 II 型错误)是相关项目中我们错误识别为不相关的。

    精确度(Precision)，表示我们发现的项目中有多少是相关的，TP/(TP+ FP)。
    召回率(Recall)，表示相关的项目中我们发现了多少，TP/(TP+FN)。
    F-度量值(F-Measure)(或 F-得分，F-Score)，组合精确度和召回率为一个单独的得分，
    被定义为精确度和召回率的调和平均数(2 × Precision × Recall)/(Precision + Recall)

    交叉检验 原始语料分成n份 使用一份做测试集 其他做训练集

自动生成分类模型 分类方法
    决策树
        缺点：在训练树的低节点，可用的训练数据量可能会变得非常小。因此，这些较低的决策节点可能过拟合训练集
        解决：当训练数据量变得太小时停止分裂节点   长出一个完整的决策树，但随后进行剪枝剪去在开发测试集上不能提高性能的决策节点
        缺点：它们强迫特征按照一个特定的顺序进行检查，即使特征可能是相对独立的
    朴素贝叶斯分类器
    最大熵分类器
"""
