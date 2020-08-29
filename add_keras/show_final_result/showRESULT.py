import matplotlib
matplotlib.use('Agg')


import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse
import matplotlib 
import numpy as np
import os
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--spec', type=str)
args = parser.parse_args()
matplotlib.rcParams.update({'font.size': 42})

#matplotlib setup
matplotlib.rcParams['agg.path.chunksize'] = 10000
file_checkpoint = args.file + 'dqn_WHG_log.json'
print(file_checkpoint)

with open(file_checkpoint, 'r') as data_json:
    data = json.load(data_json)

data = pd.DataFrame(data)
print(data.head())

features = data.columns
try:
    os.mkdir(args.file + 'dqn_result')
except:
    pass
if not args.spec:
    col = 3
    row = 3
    W = 10
    H = 10
    fig = plt.figure(figsize=(50,50))
    for i in range(1, col*row+1):
        ax = fig.add_subplot(row, col, i)
        ax.set_title(features[i-1])
        fig.plot(data[features[i-1]])
    fig.savefig(args.file + 'dqn_result.png')
    plt.show()
elif args.spec=='list':
    feature = ['loss', 'episode_reward', 'mean_q', 'nb_episode_steps']
    for fea in feature:
        ##
        #fig, ax = plt.subplots()
        plt.figure(figsize=(30,20))
        size_avg = len(data[fea])
        featrue_data = np.array(data[fea])
        print('{} is running'.format(fea))
        avg_data = np.array(np.zeros(size_avg))
        for i in tqdm(range(len(featrue_data))):
             avg_data[i] = np.nanmean(featrue_data[max(0, i-10000):i])
        plt.ylabel(fea)
        plt.plot(avg_data, linewidth=8)
        plt.savefig(args.file + 'dqn_result/{}.png'.format(fea))
        plt.clf()
        print('{} done'.format(fea))
    print('DONE!!!')

elif args.spec:   #Official
    size_avg = len(data[args.spec])
    reward_data = np.array(data[args.spec])
    avg_data = np.array(np.zeros(size_avg))
    print(reward_data)
    print(len(reward_data))
    for i in range(len(reward_data)):
         avg_data[i] = np.nanmean(reward_data[max(0, i-10000):i])
    #print(avg_data)
    plt.ylabel(args.spec)
    plt.plot(avg_data)
    plt.savefig(args.file + 'dqn_result/{}.png'.format(args.spec))
    plt.show()

#elif args.spec=='avg':
#    size_avg = len(data['episode_reward'])
#    reward_data = np.array(data['episode_reward'])
#    avg_data = np.array(np.zeros(size_avg))
#    for i in range(size_avg):
#         avg_data[i] = np.mean(reward_data[max(0, i-100):i])
#    plt.ylabel('avg_reward')
#    plt.plot(avg_data)
#    plt.savefig(args.file + 'dqn_result/avg.png')
#    plt.show()
#else:
#    plt.ylabel(args.spec, fontsize=18)
#    plt.plot(data[args.spec], linestyle='-')
#    plt.savefig(args.file + 'dqn_result/{}.png'.format(args.spec))
#    plt.show()
