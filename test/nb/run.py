import re
import random
from nb.core import *


def load_data():
    data_list=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 is abusive, 0 not
    data_class = [0, 1, 0, 1, 0, 1]
    return data_list, data_class


def text_parse(input_text):
    tokens = re.split(r'[\W0-9]*', input_text)
    return [token.lower() for token in tokens if len(token) > 0]

if __name__ == '__main__':
    vocab_list = []
    label_list = []
    # read the text
    for i in range(1, 26):
        text_tokens = text_parse(open("email/spam/%d.txt" % i, encoding="gbk").read())
        vocab_list.append(text_tokens)
        label_list.append(1)
        text_tokens = text_parse(open("email/ham/%d.txt" % i, encoding="ISO-8859-1").read())
        vocab_list.append(text_tokens)
        label_list.append(0)
    # select 10 row as training data, the others as validate data
    train_vocab_list =[]
    train_label_list = []
    for i in range(10):
        random_index = int(random.uniform(0, len(label_list)))
        train_vocab_list.append(vocab_list[random_index])
        train_label_list.append(label_list[random_index])
        del(vocab_list[random_index])
        del(label_list[random_index])
    # naive bayes
    vocab_nb = VocabNB()
    # train the example
    # post_list, class_vec = load_data()
    # vocab_nb.add_train_vocabs(post_list, class_vec)
    vocab_nb.add_train_vocabs(train_vocab_list, train_label_list)
    vocab_nb.train_nb()
    # classify
    total = 40
    right = 0
    for i in range(len(label_list)):
        should_be = label_list[i]
        classify_is = vocab_nb.classify(vocab_list[i])
        print("label should be %d, classify result is %d" % (should_be, classify_is))
        if should_be == classify_is:
            right += 1
    print("total correct rate is %f" % (right /float(total)))