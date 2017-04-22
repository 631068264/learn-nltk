#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/4/21 17:52
@annotation = ''
"""
import nltk
import requests

"""
分词 tokenization
"""

url = "http://www.gutenberg.org/files/2554/2554.txt"
response = requests.get(url=url)
response.encoding = "utf8"
raw = response.text
print(raw[:75])
# tokenization
tokens = nltk.word_tokenize(raw)
print(tokens[:10])

text = nltk.Text(tokens)
print(text[1020:1060])


def file_wr():
    with open("sdf.txt", "w", encoding="utf8") as f:
        f.write("sdfdfa方法对方水电费是")
    with open("sdf.txt", "r", encoding="utf8") as f:
        for line in f.readlines():
            print(line)


def find_all():
    from nltk.corpus import gutenberg, nps_chat

    """搜索已分词文本"""
    moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
    moby.findall(r"<a> (<.*>) <man>")
    chat = nltk.Text(nps_chat.words())
    chat.findall(r"<.*> <.*> <bro>")
    chat.findall(r"<l.*>{3,}")
