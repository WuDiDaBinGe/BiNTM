# -*- coding: utf-8 -*-
# @Time    : 2021/7/25 下午7:46
# @Author  : WuDiDaBinGe
# @FileName: caculate_coherence.py
# @Software: PyCharm
import os
import subprocess
import threading
import time

import chardet
from jpype import *

import numpy as np
from palmettopy.palmetto import Palmetto


def prcess_result(result):
    result = result.strip().split('\n')
    res = 0
    for line in result:
        res_list = line.strip().split('\t')
        if len(res_list) == 3:
            res += float(res_list[1])
    return res


def calculate_coherence_jar(topic_file, ret, coherence_type):
    command = ["java", "-jar", "/home/yxb/MyCoding/Palmetto/palmetto-0.1.0-jar-with-dependencies.jar",
               "/home/yxb/MyCoding/Palmetto/index/wikipedia_bd",
               coherence_type, topic_file]
    stdout, stderr = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    encoding = chardet.detect(stdout)["encoding"]
    result = stdout.decode(encoding)
    ret[coherence_type] = prcess_result(result)
    return


# def calculate_coherence_webserve(word_list, ret, coherence_type):
#     palmetto = Palmetto(palmetto_uri='http://localhost:7777/palmetto-webapp/service/', timeout=200)
#     result = []
#     for words in word_list:
#         result.append(palmetto.get_coherence(words, coherence_type=coherence_type))
#     ret[coherence_type] = result
#     return


# def get_coherence(topic_file):
#     thread_pool = []
#     ret = {'ca': 0, 'cp': 0, 'npmi': 0}
#     th_ca = threading.Thread(target=calculate_coherence_webserve, args=[topic_file, ret, 'ca'], name='th_ca')
#     th_cp = threading.Thread(target=calculate_coherence_webserve, args=[topic_file, ret, 'cp'], name='th_cp')
#     th_npmi = threading.Thread(target=calculate_coherence_webserve, args=[topic_file, ret, 'npmi'], name='th_npmi')
#     thread_pool.append(th_ca)
#     thread_pool.append(th_cp)
#     thread_pool.append(th_npmi)
#     start = time.time()
#     for th in thread_pool:
#         th.start()
#     for th in thread_pool:
#         th.join()
#     end = time.time()
#     print(f"多线程：{end - start}")
#     return np.mean(ret['ca']), np.mean(ret['cp']), np.mean(ret['npmi'])

def write_file(topic_words, filename):
    words = '\n'.join([" ".join(lst) for lst in topic_words])
    f = open(filename, "w")
    f.write(words)


def get_coherence_by_local_jar(topic_words):
    filename = "topic_words"
    write_file(topic_words, filename)
    thread_pool = []
    ret = {'C_A': 0, 'C_P': 0, 'NPMI': 0}
    th_ca = threading.Thread(target=calculate_coherence_jar, args=[filename, ret, 'C_A'], name='th_ca')
    th_cp = threading.Thread(target=calculate_coherence_jar, args=[filename, ret, 'C_P'], name='th_cp')
    th_npmi = threading.Thread(target=calculate_coherence_jar, args=[filename, ret, 'NPMI'], name='th_npmi')
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
    return ret['C_A'], ret['C_P'], ret['NPMI']


if __name__ == '__main__':
    words_topic = [
        ['car', 'anyone', 'bike', 'new', 'good', 'buy', 'engine', 'oil', 'ride', 'dod'],
        ['game', 'player', 'team', 'play', 'go', 'hockey', 'win', 'nhl', 'score', 'season'],
        ['msg', 'doctor', 'food', 'patient', 'gordon', 'bank', 'disease', 'go', 'treatment', 'surrender'],
        ['window', 'thanks', 'anyone', 'advance', 'display', 'color', 'screen', 'run', 'look', 'appreciate'],
        ['card', 'monitor', 'video', 'mode', 'driver', 'vga', 'color', 'mouse', 'chip', 'speed'],
        ['israel', 'israeli', 'arab', 'jews', 'bank', 'arabs', 'lebanese', 'gordon', 'lebanon', 'research'],
        ['space', 'car', 'launch', 'cost', 'nasa', 'orbit', 'moon', 'shuttle', 'price', 'look'],
    ]
    start = time.time()
    print(get_coherence_by_local_jar(words_topic))
    end = time.time()
    print("Time:" + str(end - start))
