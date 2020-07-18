import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(pd.read_csv('data_result.csv'))
#plt.plot('Reward', data = data)
#plt.plot('Max', data = data)
plt.plot('Avg Reward', data = data)
plt.show()

