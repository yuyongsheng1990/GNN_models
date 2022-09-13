# @Time : 2022/9/13 11:24
# @Author : yysgz
# @File : tri-party deep network representation.py
# @Project : tri-party deep network representation.ipynb
# @Description : 复现tri-party deep network representation算法；github address: https://github.com/GRAND-Lab/TriDNR
# @packages: gensim==3.8.3; scikit-learn==1.1.2; numpy==1.23.3; pandas==1.4.4; scipy==1.9.1

# ------------------------------------Networkutils--------------------------------------------------
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from random import shuffle
from deepwalk import graph
import gensim
import random
import gensim.utils as ut

NetworkSentence = namedtuple('NetworkSentence', 'words tags labels index')
Result = namedtuple('Result', 'alg trainsize acc macro_f1 micro_f1')
AlgResult = namedtuple('AlgResult', 'alg trainsize numfeature mean std')


def readNetworkData(dir, stemmer=0):  # dir, directory of network dataset
    allindex = {}
    alldocs = []
    labelset = set()
    with open(dir + '/docs.txt', 'r', encoding='utf-8') as f1, open(dir + '/labels.txt', 'r', encoding='utf-8') as f2:
        for l1 in f1:
            #             tokens = ut.to_unicode(l1.lower()).split()
            if stemmer == 1:
                l1 = gensim.parsing.stem_text(l1)  # step_text=lower() + step()
            else:
                l1 = l1.lower()
            tokens = ut.to_unicode(l1).split()  # to_unicode()转换成unicode编码

            words = tokens[1:]  # extract texts of document
            tags = [tokens[0]]  # ID of each document, for doc2vec model
            index = len(alldocs)
            allindex[tokens[0]] = index  # A mapping from documentID to index, start from 0

            l2 = f2.readline()
            tokens2 = gensim.utils.to_unicode(l2).split()
            labels = tokens2[1]  # class label
            labelset.add(labels)
            alldocs.append(NetworkSentence(words, tags, labels, index))
    return alldocs, allindex, list(labelset)


def trainDoc2Vec(doc_list=None, buildvoc=1, passes=20, dm=0, size=100, dm_mean=0, window=5,
                 hs=1, negative=5, min_count=1, workers=4):
    model = Doc2Vec(dm=dm, size=size, dm_mean=dm_mean, window=window, hs=hs, negative=negative,
                    min_count=min_count, workers=workers)  # PV-DBOW
    if buildvoc == 1:
        print('Building Vocabulary')
        model.build_vocab(doc_list)  # build vocabulate with words + nodeID

    for epoch in range(passes):
        print('Iteration %d ....' % epoch)
        shuffle(doc_list)  # shuffling gets best results
        model.train(doc_list, total_examples=model.corpus_count, epochs=model.epochs)

    return model

def trainWord2Vec(doc_list=None, buildvoc=1, passes=20, sg=1, size=100, dm_mean=0, window=5, hs=1,negative=5,
                 min_count=1, workers=4):
    model = Word2Vec(size=size, sg=sg, window=window, hs=hs, negative=negative, min_count=min_count, workers=workers)
    if buildvoc == 1:
        print('Building Vocabulary')
        model.build_vocab(doc_list)  # build vocabulate with words + nodeID
    for epoch in range(passes):
        print('Iteration %d ...' % epoch)
        shuffle(doc_list)
        model.train(doc_list, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def getdeepwalks(directory, number_walks=50, walk_length=10, seed=1):
    Graph = graph.load_adjacencylist(directory + '/adjedges.txt')
    print('Number of nodes: {}'.format(len(Graph.nodes())))
    num_walks = len(Graph.nodes()) * number_walks
    print('Number of walks: {}'.format(num_walks))

    print('Walking...')
    walks = graph.build_deepwalk_corpus(Graph, num_paths=number_walks, path_length=walk_length, alpha=0,
                                        rand=random.Random(seed))
    networksentence = []
    raw_walks = []
    for i, x in enumerate(walks):
        sentence = [gensim.utils.to_unicode(str(t)) for t in x]
        s = NetworkSentence(sentence, [sentence[0]], None, i)  # label information is not used by random walk
        networksentence.append(s)
        raw_walks.append(sentence)
    return raw_walks, networksentence

# -----------------------------------Evaluation-------------------------------------------------
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

from gensim.models.doc2vec import Doc2Vec


def evaluation(train_vec, test_vec, train_y, test_y, classifierStr='SVM', normalize=0):
    if classifierStr == 'KNN':
        print('Training NN classifier...')
        classifier = KNeighborsClassifier(n_neighbors=1)
    else:
        print('Training SVM classifier...')
        classifier = LinearSVC()
    if normalize == 1:
        print('Normalize data')
        allvec = list(train_vec)
        allvec.extend(test_vec)
        allvec_normalized = preprocessing.normalize(allvec, norm='l2', axis=1)
        train_vec = allvec_normalized[0:len(train_y)]
        test_vec = allvec_normalized[len(train_y):]

    # training
    classifier.fit(train_vec, train_y)
    y_pred = classifier.predict(test_vec)
    cm = confusion_matrix(test_y, y_pred)  # 混淆矩阵
    print(cm)
    acc = accuracy_score(test_y, y_pred)
    print(acc)
    macro_f1 = f1_score(test_y, y_pred, pos_label=None, average='macro')
    micro_f1 = f1_score(test_y, y_pred, pos_label=None, average='micro')

    percent = len(train_y) * 1.0 / (len(train_y) + len(test_y))
    print('Classification method:' + classifierStr + '(train, test, Training_percent): (%d, %d, %f)' %
          (len(train_y), len(test_y), percent))
    print('Classification Accuracy=%f, macro_f1=%f, micro_f1=%f' % (acc, macro_f1, micro_f1))
    # print(metrics.classification_report(test_y, y_pred))
    return acc, macro_f1, micro_f1


def evaluationEmbedModelFromTrainTest(model, train, test, classifierStr='SVM', normalize=0):
    if isinstance(model, Doc2Vec):
        # model.docvecs函数生成doc2vec向量
        train_vecs = [model.docvecs[doc.tags[0]] for doc in train]
        test_vecs = [model.docvecs[doc.tags[0]] for doc in test]
    else:  # word2vec model 生成向量
        train_vecs = [model.wv.word_vec(doc.tags[0]) for doc in train]
        test_vecs = [model.wv.word_vec(doc.tags[0]) for doc in test]
    train_y = [doc.labels for doc in train]
    test_y = [doc.labels for doc in test]
    print('train_y: , test_y: ', len(train_y), len(test_y))
    acc, macro_f1, micro_f1 = evaluation(train_vecs, test_vecs, train_y, test_y, classifierStr, normalize)

    return acc, macro_f1, micro_f1

# ---------------------------------Tri-party DNR--------------------------------------------
from sklearn.model_selection import train_test_split

from gensim.models.doc2vec import Doc2Vec
from random import shuffle


class TriDNR:
    '''
    Tri-party Deep Network Representation, IJCAI-2016
    Read data from a from a directory which contains text, label, structure information, and initialize the TriDNR from
    Doc2Vec and DeepWalk Models, then iteratively update the model with text, label, and structure information.
    'directory'
        docs.txt -- text document for each node, one line for one node
        labels.txt -- class label for each node, noe line for one node
        adjedges.txt -- edge list for each node, one line for one node
    'train_size': percentage of training data in range 0-1, if train_size==0, it becomes pure unsupervised network representation
    'text_weight': weights for the text information, 0-1
    'size': the dimensionality of the feature vectors.
    'dm': defines doc2vec the training algorithm. dm=1, PV_DM; otherwise, PV-DBOW.
    'min_count': minimum number of counts for words.
    '''

    def __init__(self, directory=None, train_size=0.3, textweight=0.8, size=300, seed=1, workers=1, passes=10, dm=0,
                 min_count=3):
        # Read the data
        alldocs, docindex, classlabels = readNetworkData(directory)
        print('%d document, %d classes, training ratio=%f' % (len(alldocs), len(classlabels), train_size))

        # Initialize Doc2Vec
        if train_size > 0:  # label information is available for learning
            print('Adding Label Information')
            train, test = train_test_split(alldocs, train_size=train_size, random_state=seed)
            '''
            add supervised information to training data, use label information for learning.
            Specifically, the doc2vec algorithm used the tags information as document IDs, and learn a vector
                representation for each tag(ID),.
            We add the class label into tags, so each class will acts as a ID and is used to learn the latent representation.
            '''
            alldata = train[:]
            for x in alldata:
                x.tags.append('Label' + x.labels)
            alldata.extend(test)
        else:  # no label information is availabel, pure unsupervised learning
            alldata = alldocs[:]

        d2v = trainDoc2Vec(alldata, workers=workers, size=size, dm=dm, passes=passes, min_count=min_count)

        raw_walks, netwalks = getdeepwalks(directory, number_walks=20, walk_length=8)
        w2v = trainWord2Vec(raw_walks, buildvoc=1, passes=passes, size=size, workers=workers)
        if train_size > 0:  # print out the initial results
            print('initialize Doc2Vec Model with supervised Information...')
            evaluationEmbedModelFromTrainTest(d2v, train, test, classifierStr='SVM')
            print('Initialize DeepWalk model')
            evaluationEmbedModelFromTrainTest(w2v, train, test, classifierStr='SVM')

        self.d2v = d2v
        self.w2v = w2v
        self.doctags = [doc.tags[0] for doc in alldocs]

        self.train(d2v, w2v, directory, alldata, passes=passes, weight=textweight)

        if textweight > 0.5:
            self.model = d2v
        else:
            self.model = w2v

    def setWeights(self, d2v_model, w2v_model, weight=1):
        if isinstance(d2v_model, Doc2Vec):
            print('Copy weights from Doc2Vec to Word2Vec')
            keys = w2v_model.wv.vocab.keys()
            for key in keys:
                if key not in self.doctags:
                    continue
                w2v_index = w2v_model.wv.vocab[key].index  # word2Vec index
                w2v_model.wv.syn0[w2v_index] = (1-weight) * w2v_model.wv.syn0[w2v_index] + \
                                weight * d2v_model.docvecs[key]

    def train(self, d2v, w2v, directory, alldata, passes=10, weight=0.9):
        raw_walks, walks = getdeepwalks(directory, number_walks=20, walk_length=10)
        for i in range(passes):
            print('Iterative Runing %d' % i)
            self.setWeights(d2v, w2v, weight=weight)
            # Train Word2Vec
            shuffle(raw_walks)
            print('Update W2V...')
            w2v.train(raw_walks, total_examples=w2v.corpus_count, epochs=w2v.epochs)
            self.setWeights(w2v, d2v, weight=(1 - weight))

            print('Update D2V...')
            shuffle(alldata)  # shuffling to get best results
            d2v.train(alldata, total_examples=d2v.corpus_count, epochs=d2v.epochs)

# -------------------------------------demo---------------------------------------------------
from sklearn.model_selection import train_test_split
import numpy as np

numFea = 100
cores = 4
train_size = 0.2  # percentage of training samples
random_state = 2
dm = 0
passes = 20

directory = 'tri-party data/M10'
alldocs, allsentence, classlabels = readNetworkData(directory)
print('%d document' % len(alldocs))
print('%d classes' % len(classlabels))
doc_list = alldocs[:]  # for reshuffling pass

train, test = train_test_split(doc_list, train_size=train_size, random_state=random_state)

# baselin 1, Doc2Vec model(PV-DM)
print('#############')
print('baseline 1, Doc2Vec Model dm=%d' % dm)
doc2vec_model = trainDoc2Vec(doc_list, workers=cores, size=numFea, dm=dm, passes=passes, min_count=3)

print('Classification Performance on Doc2Vec Model')
doc2vec_acc, doc2vec_macro_f1, doc2vec_micro_f1 = \
    evaluationEmbedModelFromTrainTest(doc2vec_model, train, test, classifierStr='SVM')
print('#############')

# baseline 2, DeepWalk model
print('#############')
print('baseline 2, DeepWalk model')
raw_walks, netwalks = getdeepwalks(directory, number_walks=20, walk_length=8)
deepwalk_model = trainWord2Vec(raw_walks, buildvoc=1, sg=1, passes=passes, size=numFea, workers=cores)
print('classification performance on DeepWalk model')
doc2vec_acc, doc2vec_macro_f1, doc2vec_micro_f1 = \
    evaluationEmbedModelFromTrainTest(deepwalk_model, train, test, classifierStr='SVM')
print('##############')

# baseline 3, D2V+DW
print('##############')
print('baseline 3, simple combination of DeepWalk + Doc2Vec')
d2v_train_vecs = [doc2vec_model.docvecs[doc.tags[0]] for doc in train]
d2v_test_vecs = [doc2vec_model.docvecs[doc.tags[0]] for doc in test]

dw_train_vecs = [deepwalk_model.wv.word_vec(doc.tags[0]) for doc in train]
dw_test_vecs = [deepwalk_model.wv.word_vec(doc.tags[0]) for doc in test]

train_y = [doc.labels for doc in train]
test_y = [doc.labels for doc in test]

# concanate two vectors
train_vecs = [np.append(l, dw_train_vecs[i]) for i,l in enumerate(d2v_train_vecs)]
test_vecs = [np.append(l, dw_test_vecs[i]) for i,l in enumerate(d2v_test_vecs)]

print('train_y: , test_y: ', len(train_y), len(test_y))
print('Classifcation Performance on Doc2Vec + DeepWalk')

acc, macro_f1, micro_f1 = evaluation(train_vecs, test_vecs, train_y, test_y, classifierStr='SVM')

# tri-party dnr model

tridnr_model = TriDNR(directory, size=numFea, dm=0, textweight=0.8, train_size=train_size, seed=random_state,
                     passes=10)
evaluationEmbedModelFromTrainTest(tridnr_model.model, train, test, classifierStr='SVM')
