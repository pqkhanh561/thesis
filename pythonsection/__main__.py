import subprocess
import time
from env import env


if __name__ == "__main__":
    i = 0
    dem =0
    state = 0
    e=env()
    while True:
        e.step(i)
        i = i + 1
        dem = dem + 1
        if dem>=1: i=i-1
        if dem==100:
            dem=0
            i=i+1
        if i==4:
            i=0
