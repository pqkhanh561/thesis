import subprocess
import time



if __name__ == "__main__":
    i = 0
    while True:
        if i == 0:
            subprocess.call('cp -r ../../actionl.txt ../action.txt', shell=True)
        if i==1:
            subprocess.call('cp -r ../../actiond.txt ../action.txt', shell=True)
        if i==2:
            subprocess.call('cp -r ../../actionr.txt ../action.txt', shell=True)
        if i==3:
            subprocess.call('cp -r ../../actionu.txt ../action.txt', shell=True)
        i = i + 1
        if i ==4:
            i=0
        time.sleep(0.005)
