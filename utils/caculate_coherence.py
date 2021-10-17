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
    count = 0
    for line in result:
        res_list = line.strip().split('\t')
        if len(res_list) == 3:
            res += float(res_list[1])
            count += 1
    return res / count


def calculate_coherence_jar(topic_file, ret, coherence_type):
    command = ["java", "-jar", "/home/yxb/MyCoding/Palmetto/palmetto-0.1.0-jar-with-dependencies.jar",
               "/home/yxb/MyCoding/Palmetto/index/wikipedia_bd",
               coherence_type, topic_file]
    stdout, stderr = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    encoding = chardet.detect(stdout)["encoding"]
    result = stdout.decode(encoding)
    if coherence_type == "NPMI":
        print(result)
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


def get_coherence_by_local_jar(topic_words, date, task_name):
    filename = f"topic_words_{task_name}_{len(topic_words)}_{date}"
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
        ['israel', 'lebanese', 'lebanon', 'israeli', 'civilian', 'territory', 'arab', 'jews', 'palestinian', 'make'],
        ['driver', 'card', 'gateway', 'latest', 'video', 'run', 'load', 'mouse', 'mode', 'windows'],
        ['jesus', 'sin', 'christian', 'bible', 'christ', 'god', 'love', 'use', 'make', 'lord'],
        ['space', 'nasa', 'billion', 'launch', 'high', 'rocket', 'fly', 'solar', 'moon', 'station'],
        ['simm', 'nec', 'apple', 'keyboard', 'like', 'pin', 'mac', 'suggestion', 'nice', 'advice'],
        ['oil', 'spot', 'bike', 'button', 'air', 'helmet', 'reliable', 'clean', 'plug', 'put'],
        ['church', 'christian', 'bible', 'scientific', 'revelation', 'book', 'part', 'christianity', 'point', 'see'],
        ['game', 'team', 'score', 'cup', 'play', 'ice', 'hockey', 'season', 'make', 'see'],
        ['koresh', 'batf', 'tear', 'compound', 'die', 'fire', 'gas', 'cs', 'agent', 'see'],
        ['car', 'turbo', 'model', 'sport', 'road', 'sit', 'engine', 'handle', 'drive', 'brake'],
        ['shipping', 'original', 'cable', 'offer', 'manual', 'sell', 'floppy', 'please', 'hp', 'new'],
        ['gun', 'criminal', 'firearm', 'violent', 'ban', 'crime', 'make', 'case', 'people', 'see'],
        ['player', 'league', 'play', 'career', 'defensive', 'nhl', 'baseball', 'hockey', 'average', 'roger'],
        ['problem', 'fix', 'compile', 'null', 'patch', 'error', 'bug', 'turn', 'program', 'file'],
        ['armenians', 'armenian', 'turks', 'armenia', 'turkish', 'serve', 'turkey', 'population', 'muslim', 'soviet'],
        ['key', 'algorithm', 'chip', 'clipper', 'secure', 'escrow', 'trust', 'encryption', 'government', 'secret'],
        ['clock', 'slow', 'speed', 'faster', 'cpu', 'mhz', 'fast', 'processor', 'bus', 'handle'],
        ['find', 'hello', 'know', 'please', 'vga', 'file', 'driver', 'program', 'update', 'anyone'],
        ['bike', 'dog', 'dod', 'ride', 'honda', 'make', 'like', 'drink', 'roll', 'disclaimer'],
        ['disease', 'drug', 'much', 'money', 'people', 'tax', 'pay', 'oh', 'take', 'since'],
    ]
    # for line in words_topic:
    #     print(" ".join(line))
    start = time.time()
    print(get_coherence_by_local_jar(words_topic, time.strftime("%Y-%m-%d-%H-%M", time.localtime()), 'test'))
    end = time.time()
    print("Time:" + str(end - start))
