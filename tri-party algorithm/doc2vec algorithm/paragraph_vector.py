# -*- coding: utf-8 -*-
# @Time : 2022/9/12 15:55
# @Author : yysgz
# @File : paragraph_vector.py
# @Project : test.py
# @Description : 复现paragraph vector

from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess
from smart_open import open

file_name = './data/pv_dm_data/proasmdataset.txt'
train_vec = 'proasmdatasetVec.txt.model'

def read_corpus(filename, tokens_only=False):
    with open(filename, encoding='utf-8') as f:
        for i,line in enumerate(f):
            tokens = simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                yield TaggedDocument(tokens, [i])

train_corpus = list(read_corpus(file_name))
test_corpus = list(read_corpus(file_name, tokens_only=True))

def train(ftrain):
    # 实例化Doc2Vec模型
    model = Doc2Vec(vector_size=100, window=3, cbow_mean=1, min_count=1)
    # 更新现有的word2vec模型
    model.build_vocab(ftrain)  # 使用数据建立单词表
    model.train(ftrain, total_examples=model.corpus_count, epochs=10)  # 训练模型，更新模型参数
    model.save(train_vec)
    return model

model_dm = train(train_corpus)

# 模型训练完成后，可以用来生成一段文本的paragraph vector。
test_document = ['only', 'you', 'can', 'prevent', 'forest', 'fires']
test_vector = model_dm.infer_vector(test_document)
print(test_vector)
