# -*- coding: utf-8 -*-
# @Time    : 2021/7/25 下午7:46
# @Author  : WuDiDaBinGe
# @FileName: caculate_coherence.py
# @Software: PyCharm
import threading
import time
import numpy as np

from palmettopy.palmetto import Palmetto

palmetto = Palmetto(palmetto_uri='http://localhost:7777/palmetto-webapp/service/', timeout=200)


def calculate_coherence(word_list, ret, coherence_type):
    result = []
    for words in word_list:
        result.append(palmetto.get_coherence(words, coherence_type=coherence_type))
    ret[coherence_type] = result
    return


def get_coherence(topic_words):
    thread_pool = []
    ret = {'ca': 0, 'cp': 0, 'npmi': 0}
    th_ca = threading.Thread(target=calculate_coherence, args=[topic_words, ret, 'ca'], name='th_ca')
    th_cp = threading.Thread(target=calculate_coherence, args=[topic_words, ret, 'cp'], name='th_cp')
    th_npmi = threading.Thread(target=calculate_coherence, args=[topic_words, ret, 'npmi'], name='th_npmi')
    thread_pool.append(th_ca)
    thread_pool.append(th_cp)
    thread_pool.append(th_npmi)
    start = time.time()
    for th in thread_pool:
        th.start()
    for th in thread_pool:
        th.join()
    end = time.time()
    print(f"多线程：{end - start}")
    return np.mean(ret['ca']), np.mean(ret['cp']), np.mean(ret['npmi'])
