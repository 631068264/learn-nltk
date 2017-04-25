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
from nltk.corpus import names

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


# classify()
better_classify()
