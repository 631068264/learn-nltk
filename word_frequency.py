#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/4/18 17:52
@annotation = 词频 词长
"""

import nltk
from nltk.book import *

s = "aa啦啦方"
d = "aabbbsdfdf"


def lexical_diversity():
    """每个字平均被使用次数"""
    run = lambda x: len(x) / len(set(x))
    print(run(s))


def percentage():
    """特定的词在文本中占据的百分比"""

    def run(key, text):
        return 100 * text.count(key) / len(text)

    print(run("方", s))


def run(text):
    return nltk.FreqDist(text)


def word_frequency():
    """词频"""
    fdist = run("在宗教关系紧张和民间排外情绪加剧下，印尼雅加达省（雅京特区）进行新一届省长选举。")
    print(fdist[""])
    print(fdist.keys())
    """只出现一次"""
    print(fdist.hapaxes())
    print(fdist.most_common(3))
    # fdist.plot()


def count_length():
    """词长的分布"""
    fdist = run(([len(w) for w in text1]))
    print(type(fdist))
    print(fdist.keys())
    print(fdist.most_common())
    """最频繁的词长度是 3"""
    print(fdist.max())
    """长度为 3 的词有 50,000 多个"""
    print(fdist[3])
    print(fdist.items())
    """"约占书中全部词汇的 20%"""
    print(fdist.freq(3) * 100)


count_length()

"""
dist = FreqDist(samples) 创建包含给定样本的频率分布
fdist.inc(sample) 增加样本
fdist['monstrous' ] 计数给定样本出现的次数
fdist.freq('monstrous ' ) 给定样本的频率
fdist.N() 样本总数
fdist.keys( ) 以频率递减顺序排序的样本链表
for sample in fdist: 以频率递减的顺序遍历样本
fdist.max() 数值最大的样本
fdist.tabulate() 绘制频率分布表
fdist.pl ot() 绘制频率分布图
fdist.plot(cumulative=True) 绘制累积频率分布图
fdist1 < fdist2 测试样本在 fdist1 中出现的频率是否小于 fdist2


Example	Description
fdist = FreqDist(samples)	create a frequency distribution containing the given samples
fdist[sample] += 1	increment the count for this sample
fdist['monstrous']	count of the number of times a given sample occurred
fdist.freq('monstrous')	frequency of a given sample
fdist.N()	total number of samples
fdist.most_common(n)	the n most common samples and their frequencies
for sample in fdist:	iterate over the samples
fdist.max()	sample with the greatest count
fdist.tabulate()	tabulate the frequency distribution
fdist.plot()	graphical plot of the frequency distribution
fdist.plot(cumulative=True)	cumulative plot of the frequency distribution
fdist1 |= fdist2	update fdist1 with counts from fdist2
fdist1 < fdist2	test if samples in fdist1 occur less frequently than in fdist2

"""
