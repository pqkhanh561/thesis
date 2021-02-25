# Exploiting RL to find optimal strategy in dynamic maps  
Try to solve The World's Hardest Game by using Vanilla Q-learning and some config. Best result is 55,8% win the new game that we created.

## Structure
1. add_keras_rl: using keras-rl2 all python include game

2. old_implement: using tf 1.4 and using old java map

3. statistic_WHG: exploration the map

4. tex: latex report in vietnamese
## To train 
```
cd add_keras
python dqn_atari.py --mode=train --weights=... --render=0
```
## To run demo
```
cd add_keras
python dqn_atari.py --mode=test --weights=weights_filename/dqn_WHG_weights_3000000.h5f --render=1
```
The best result was trained 6M steps (need more)
Caution: run in python 3.6
