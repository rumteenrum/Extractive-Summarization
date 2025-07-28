
import os, sys, time
import argparse
from shutil import copyfile


# be careful! this one is AOE
def kill_training_processes(delay=0):
    if delay > 0:
        time.sleep(delay)

    pids = []
    lines = os.popen('ps -ef | grep python').readlines()
    for line in lines:
        fds = line.strip().split()
        if 'train.py' in line or 'multiprocessing.spawn' in line or '<defunct>' in line:
            print(line)
            print(fds[1])
            pids.append(fds[1])
    
    for pid in pids:
        cmd = 'kill -9 {}'.format(pid)
        print(cmd)
        os.system(cmd)
    time.sleep(5)

    os.system('nvidia-smi')


if __name__ == '__main__':
    kill_training_processes()


