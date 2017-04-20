#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/4/20 17:47
@annotation = 词语搭配和双连词
"""
import nltk
from nltk.text import Text
from nltk.book import *

d = nltk.bigrams(['more', 'is', 'said', 'than', 'done'])
print(list(d))

"""频繁出现的双连词 体现文本的风格"""
print(text4.collocations())
