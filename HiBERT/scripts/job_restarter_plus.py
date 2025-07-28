
import os, sys, time, re
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


def is_backup(traindir):
    import filecmp
    LOG = 'log.txt'
    logfile = os.path.join(traindir, LOG)
    for f in os.listdir(traindir):
        if f.startswith(LOG) and f != LOG:
            bak_log_file = os.path.join(traindir, f)
            if filecmp.cmp(logfile, bak_log_file):
                return True
    return False


def backup_log(traindir):
    logfile = os.path.join(traindir, 'log.txt')
    if os.path.exists(logfile):
        # check if the logfile has been backup
        if is_backup(traindir):
            print('has backup already')
            return

        # backup file
        lid = 1
        while True:
            new_logfile = '{}.{}'.format(logfile, lid)
            if not os.path.exists(new_logfile):
                copyfile(logfile, new_logfile)
                break
            lid += 1


def run_job(script):
    cmd = 'sh {} &'.format(script)
    print(cmd)
    os.system(cmd)


def get_finished_ckpt_old(traindir):
    # checkpoint1.pt
    epoch = 0
    while True:
        ckpt = os.path.join(traindir, 'checkpoint{}.pt'.format(epoch+1))
        if not os.path.exists(ckpt):
            break
        else:
            epoch += 1


    return epoch

def get_finished_ckpt(traindir):
    epoch = 0
    for f in os.listdir(traindir):
        mat = re.search('checkpoint(\\d+)\\.pt', f)
        if mat is None: continue
        if len(mat.group()) == len(f):
            cur_epoch = int(mat.group(1))
            epoch = max(epoch, cur_epoch)
    return epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindir', default=None)
    parser.add_argument('--script', default='/hdfs/msrlabs/xizhang/sum/scripts/pretrain_doc.3G.trep.hpara.e100.msrlabs.bs13.debug.sh')
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--delay', type=int, default=180)

    args = parser.parse_args()
    if args.traindir is None:
        args.traindir = args.script.replace('/scripts/', '/experiments/')
    if not os.path.exists(args.traindir):
        os.mkdir(args.traindir)
    print(args)

    is_first = True

    while True:
        epoch = get_finished_ckpt(args.traindir)
        print('cur epoch', epoch)
        if epoch >= args.max_epoch:
            break

        if is_first:
            delay = 0
            is_first = False
        else:
            delay = args.delay
        kill_training_processes(delay=delay)
        backup_log(args.traindir)
        run_job(args.script)
        while True:
            new_epoch = get_finished_ckpt(args.traindir)
            if new_epoch >= args.max_epoch or new_epoch >= epoch+args.interval:
                break
            time.sleep(5)


