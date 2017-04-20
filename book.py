#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/4/20 17:32
@annotation = ''
"""
from nltk.book import *

V = set(text1)
long_words = [w for w in V if len(w) > 15]
print(sorted(long_words))