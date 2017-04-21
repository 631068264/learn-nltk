#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/4/20 23:10
@annotation = 语料库
"""
import nltk
from nltk.corpus import gutenberg, brown
from nltk.text import Text


def all_corpus():
    print(gutenberg.fileids())


def get_corpus(text_name):
    return gutenberg.words(text_name)


def key_word(text, key):
    return Text(text).concordance(key)


# emma = get_corpus("austen-emma.txt")
# print(emma)
# key_result = key_word(emma, "surprize")
def iter_gutenberg():
    for fileid in gutenberg.fileids():
        # 我们没有进行过任何语言学处理的文件的内容
        num_chars = len(gutenberg.raw(fileid))

        num_words = len(gutenberg.words(fileid))
        # 文本划分成句子，其中每一个句子是一个词链表
        num_sents = len(gutenberg.sents(fileid))
        num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
        """平均词长  平均句子长度  本文中每个词出现的平均次数(我们的词汇多样性得分)"""
        print(int(num_chars / num_words), int(num_words / num_sents), int(num_words / num_vocab), fileid)


def plot():
    """条件频率分布"""
    cfd = nltk.ConditionalFreqDist(
        (genre, word)
        for genre in brown.categories()
        for word in brown.words(categories=genre))
    genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
    modals = ['can', 'could', 'may', 'might', 'must', 'will']
    cfd.tabulate(conditions=genres, samples=modals)

    # cfd = nltk.ConditionalFreqDist(
    #     (target, fileid[:4])
    #     for fileid in inaugural.fileids()
    #     for w in inaugural.words(fileid)
    #     for target in ['america', 'citizen']
    #     if w.lower().startswith(target))
    # cfd.plot()

    # languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
    # cfd = nltk.ConditionalFreqDist(
    #     (lang, len(word))
    #     for lang in languages
    #     for word in udhr.words(lang + '-Latin1'))
    # cfd.plot(cumulative=True)


def cond_freq_dict():
    """
        (条件，事件)
    """
    genre_word = [
        (genre, word)
        for genre in ['news', 'romance']
        for word in brown.words(categories=genre)
        ]
    cfd = nltk.ConditionalFreqDist(genre_word)
    print(cfd.conditions())
    print(cfd['news'])
    print(cfd['romance'])


def load_corpus():
    print(gutenberg.root)
    from nltk.corpus import PlaintextCorpusReader
    corpus_root = '/path/of/corpus'
    wordlists = PlaintextCorpusReader(corpus_root, '.*')
    wordlists.fileids()


"""
corpus.语料库

fileids()                   语料库中的文件
fileids([categories])       这些分类对应的语料库中的文件
categories()                语料库中的分类
categories([fileids])       这些文件对应的语料库中的分类
raw()                       语料库的原始内容
raw(fileids=[f1,f2, f3])    指定文件的原始内容
raw(categories=[c1,c2])     指定分类的原始内容
words()                     整个语料库中的词汇
words(fileids=[f1, f2, f3]) 指定文件中的词汇
words(categories=[c1, c2])  指定分类中的词汇
sents()                     指定分类中的句子
sents(fileids=[f1,f2, f3] ) 指定文件中的句子
sents(categori es=[c1,c2])  指定分类中的句子
abspath(fileid)             指定文件在磁盘上的位置
encoding(fileid)            文件的编码(如果知道的话)
open(fileid)                打开指定语料库文件的文件流
root                        到本地安装的语料库根目录的路径
readme()	                the contents of the README file of the corpus

"""

"""
条件频率分布

cfdist= ConditionalFreqDist(pairs)      从配对链表中创建条件频率分布
cfdist.conditions()                     将条件按字母排序
cfdist[condition]                       此条件下的频率分布
cfdist[condition][sample]               此条件下给定样本的频率
cfdist.tabulate()                       为条件频率分布制表
cfdist.tabulate(samples, conditions)    指定样本和条件限制下制表
cfdist.plot()                           为条件频率分布绘图
cfdist.plot(samples, conditions)        指定样本和条件限制下绘图
cfdist1 < cfdist2                       测试样本在cfdist1中出现次数是否小于在cfdist2中出现次 数
"""
