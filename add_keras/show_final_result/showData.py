import pandas as pd 
import json
import matplotlib.pyplot as plt

path1 = './dqn_1.json'
path2 = './dqn_2.json'
path = './dqn_WHG_log.json'

with open(path1) as f:
    data = json.load(f)

df1 = pd.DataFrame(data)
print(df1.head())
print(df1.shape)

with open(path2) as f:
    data = json.load(f)

df2 = pd.DataFrame(data)
print(df2.head())
print(df2.shape)


final_data = [df1, df2]

final_data = pd.concat(final_data, ignore_index=True)
print(final_data.head())
print(final_data.shape)
final_data.to_json(path)
#plt.plot(final_data['episode_reward'])
#plt.show()
