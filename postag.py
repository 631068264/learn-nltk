#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/4/22 12:55
@annotation = 词性标注
"""
import nltk
from nltk import word_tokenize
from nltk.corpus import brown


def pos_tag():
    text = word_tokenize("And now for something completely different")
    print(nltk.pos_tag(text))


def similar():
    # TODO:similar ?
    token = word_tokenize("And now for something completely different" * 9)
    text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
    # text = nltk.Text(token)
    text.similar('now')
    text.similar('woman')
    text.similar('bought')
    text.similar('over')
    text.similar('the')


def tag_token():
    tagged_token = nltk.tag.str2tuple('fly/NN')


brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')


def test_tag():
    """train test"""
    unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)

    tags = unigram_tagger.tag(brown_sents[2007])
    # 评价tag正确率
    print(unigram_tagger.evaluate(brown_tagged_sents), tags)


def likely_tags():
    """100 个最频繁的词"""
    fd = nltk.FreqDist(brown.words(categories='news'))
    print(brown.tagged_words(categories='news'))
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    most_freq_words = fd.most_common(100)
    # likely_tags （cond,cond下最大概率发生的事件）
    likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
    # 只使用likely_tags 导致很多词无法标注
    baseline_tagger = nltk.UnigramTagger(model=likely_tags)
    print(baseline_tagger.evaluate(brown_tagged_sents))

    # 测试标注器 出现词性为None('Only', None)
    sent = brown.sents(categories='news')[3]
    print(baseline_tagger.tag(sent))

    # 查找标注器将只存储名词以外的词的词-标记对，只要它不能给一个词分配标记，它将会调用默认标注器
    baseline_tagger = nltk.UnigramTagger(model=likely_tags,
                                         backoff=nltk.DefaultTagger('NN'))
    print(baseline_tagger.tag(sent))
    # tagger 性能提高  model规模增大 性能也会提高
    print(baseline_tagger.evaluate(brown_tagged_sents))


def traintest():
    """
    n-gram tagger 根据前n-1个上文的词性决定当前token的词性
        n 越大 同一个train中数据不存在的上下文的几率也增大，只要上文确定不了词性 后面就会崩了 导致evaluate准确率下降
        (解决精度和覆盖范围之间的权衡) 可以多个tagger组合利用backoff属性

    1-gram tagger is another term for a unigram tagger:
        i.e., the context used to tag a token is just the text of the token itself.
    2-gram taggers are also called bigram taggers, and 3-gram taggers are called trigram taggers.
    """
    size = int(len(brown_tagged_sents) * 0.9)
    train_sents = brown_tagged_sents[:size]
    test_sents = brown_tagged_sents[size:]
    # 1-gram tagger
    unigram_tagger = nltk.UnigramTagger(train_sents)
    print(unigram_tagger.evaluate(test_sents))
    # 2-gram taggers
    bigram_tagger = nltk.BigramTagger(train_sents)
    print(bigram_tagger.evaluate(test_sents))

    # Combining Taggers
    t0 = nltk.DefaultTagger('NN')
    t1 = nltk.UnigramTagger(train_sents, backoff=t0)
    # 丢弃那些只看到一次或两次的上下文
    t2 = nltk.BigramTagger(train_sents, backoff=t1, cutoff=2)
    print(t2.evaluate(test_sents))

    """
    保存训练好标注器 训练耗时
    """
    import pickle
    # output = open('t2.pkl', 'wb')
    with open('t2.pkl', 'wb') as f:
        pickle.dump(t2, f, pickle.HIGHEST_PROTOCOL)
    with open('t2.pkl', 'rb') as f:
        tagger = pickle.load(f)
    text = """The board's action shows what free enterprise
         is up against in our complex maze of regulatory laws ."""
    tokens = text.split()
    print(tagger.tag(tokens))
    # output.close()


def performance_limit():
    # n-gram tagger 测n=3
    # 3元token组 给定当前单词及其前两个标记
    cfd = nltk.ConditionalFreqDist(
        ((x[1], y[1], z[0]), z[1])
        for sent in brown_tagged_sents
        for x, y, z in nltk.trigrams(sent))
    # 词性歧义 len(cfd[c]) > 1
    ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
    print(sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N())


def brill():
    # nltk.BrillTagger()
    """
    n-gram 标注器的一个潜在的问题是它们的 n-gram 表的大小(或语言模型)。
    如果使用各 种语言技术的标注器部署在移动计算设备上，在模型大小和标注器性能之间取得平衡是很重要的。
    使用回退标注器的 n-gram 标注器可能存储 trigram 和 bigram 表，这是很大的稀疏阵列,可能有数亿条条目。

    第二个问题是关于上下文的。n-gram 标注器从前面的上下文中获得的唯一的信息是标记，虽然词本身可能是一个有用的信息源。
    n-gram 模型使用上下文中的词的其他特征为条 件是不切实际的。
    在本节中，我们考察 Brill 标注，一种归纳标注方法，它的性能很好，使用的模型只有 n-gram 标注器的很小一部分。

    Brill 标注是一种基于转换的学习，以它的发明者命名。一般的想法很简单:猜每个词的标记，然后返回和修复错误的。
    在这种方式中，Brill 标注器陆续将一个不良标注的文本 转换成一个更好的。与 n-gram 标注一样，
    这是有监督的学习方法，因为我们需要已标注的 训练数据来评估标注器的猜测是否是一个错误。
    然而，不像 n-gram 标注，它不计数观察结 果，只编制一个转换修正规则链表。

    Brill 标注的的过程通常是与绘画类比来解释的。假设我们要画一棵树，包括大树枝、 树枝、小枝、叶子和一个统一的天蓝色背景的所有细节。
    不是先画树然后尝试在空白处画蓝色，而是简单的将整个画布画成蓝色，然后通过在蓝色背景上上色“修正”树的部分。
    以同样的方式，我们可能会画一个统一的褐色的树干再回过头来用更精细的刷子画进一步的细节。
    Brill 标注使用了同样的想法:以大笔画开始，然后修复细节，一点点的细致的改变。
    """


brill()
