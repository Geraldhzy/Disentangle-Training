import argparse, time
import numpy as np
from matplotlib import pyplot as plt
import math

def read_test_data(test_file):
    fin = open(test_file, 'r')
    test_data = [line.strip().split('\t') for line in fin.readlines()]
    fin.close()
    return test_data

def read_vector(vector_file):
    fwv = open(vector_file, 'r')
    header = next(fwv)
    vector_dict = {}
    for line in fwv:
        line = line.strip().split(' ')
        vector_dict[line[0]] = np.array(list(map(float, line[1:])))
    return vector_dict

def similarity(v1, v2):
    s = np.dot(v1,v2)/(np.linalg.norm(v1)*(np.linalg.norm(v2)))
    if math.isnan(s):
        s = 0.5
    return s

def cal_sim(embed):
    sim = []
    for v1,v2 in embed:
        sim.append(similarity(v1, v2))
    return sim

def list2pair(embed_list):
    embed_pairs = []
    for i in range(int(len(embed_list)/2)):
        embed_pairs.append([embed_list[i*2], embed_list[i*2+1]])
    return embed_pairs

def predict_label(embed, threshold=0.85):
    sim = cal_sim(embed)
    predict = []
    for s in sim:
        if s >= threshold:
            predict.append(1)
        else:
            predict.append(0)
    return predict

def cal_score(predict, label):
    total = len(label)
    right,tru_pos, fal_pos, real_pos = 0, 0, 0, 0
    for i,p in enumerate(predict):
        if int(label[i]) == 1:
            real_pos += 1
        if p == int(label[i]):
            right += 1
            if p == 1:
                tru_pos += 1
        else:
            if p == 1:
                fal_pos += 1
    acc = right/total
    if tru_pos+fal_pos != 0:
        pre = tru_pos/(tru_pos+fal_pos)
    else:
        pre = 0
    rec = tru_pos/real_pos
    f1 = 2*(pre*rec)/(pre + rec)
    return acc, pre, rec, f1

def evaluate(W_val, test_tv, test_label, printout=False, savepath=None):
    max_acc = 0
    scores = []
    best_t = 0
    result_pre = 0
    result_rec = 0
    result_f1 = 0
    embed = list2pair(test_tv)
    for i in range(50):
        t = 1 - i*0.01
        predict = predict_label(embed, threshold=t)
        acc, pre, rec, f1 = cal_score(predict, test_label)
        if acc > max_acc:
            max_acc = acc
            best_t = t
            result_pre = pre
            result_rec = rec
            result_f1 = f1
        scores.append([t,acc,pre,rec,f1])
    if printout:
        print("Best T: {:.2f}".format(best_t))
        print("Accuracy: {:.2%}".format(max_acc))
        print("Precision: {:.2%}".format(result_pre))
        print("Recall: {:.2%}".format(result_rec))
        print("F1-score: {:.2%}".format(result_f1))
    if savepath:
        fout = open(savepath + '.csv', 'w')
        fout.write("Threshold,Accuracy,Precision,Recall,F1\n")
        fout.writelines([','.join([str(round(c,4)) for c in s])+'\n' for s in scores])
        fout.close()
    return max_acc

def sent2vec(sent, vecs):
    embed_dim = len(vecs['a'])
    v = np.zeros(embed_dim)
    c = 0
    for w in sent:
        if w in vecs:
            v += vecs[w]
            c += 1
    if c:
        v /= c
    return v
