#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/4/21 23:39
@annotation = 词干提取
"""
import nltk
from nltk import word_tokenize

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
    is no basis for a system of government.  Supreme executive power derives from
    a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = word_tokenize(raw)


def get_stem():
    """
    The Porter Stemmer is a good choice
        if you are indexing some texts
        and want to support search using alternative forms of words
    """
    porter = nltk.PorterStemmer()
    lancaster = nltk.LancasterStemmer()
    print([porter.stem(t) for t in tokens])
    print([lancaster.stem(t) for t in tokens])


def stem_index():
    """词干索引"""

    class IndexedText(object):
        def __init__(self, stemmer, text):
            self._text = text
            self._stemmer = stemmer
            self._index = nltk.Index((self._stem(word), i)
                                     for (i, word) in enumerate(text))

        def concordance(self, word, width=40):
            key = self._stem(word)
            wc = int(width / 4)  # words of context
            for i in self._index[key]:
                lcontext = ' '.join(self._text[i - wc:i])
                rcontext = ' '.join(self._text[i:i + wc])
                ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
                rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
                print(ldisplay, rdisplay)

        def _stem(self, word):
            return self._stemmer.stem(word).lower()

    porter = nltk.PorterStemmer()
    grail = nltk.corpus.webtext.words('grail.txt')
    print(grail)
    text = IndexedText(porter, grail)
    text.concordance('lie')


def lemmatization():
    """Lemmatization 词形归并 """
    """
    WordNet词形归并器删除词缀产生的词都是在它的字典中的词。
        这个额外的检查过程使词形归并器比刚才提到的词干提取器要慢。
    """
    wnl = nltk.WordNetLemmatizer()
    print([wnl.lemmatize(t) for t in tokens])


lemmatization()
get_stem()