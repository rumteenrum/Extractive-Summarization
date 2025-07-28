
import os, sys, argparse

def get_basedir(infile):
    for line in open(infile):
        line = line.strip()
        key = 'basedir'
        if line.startswith(key):
            pos = line.find('=')
            return line[pos+1:].strip()
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script')
    parser.add_argument('--restarter', default='job_restarter_plus.py')
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--vc', default='wu1')

    args = parser.parse_args()

    basedir = get_basedir(args.script)
    label = os.path.basename(args.script)
    script_dir = os.path.join(basedir, 'scripts')
    restarter_path = os.path.join(script_dir, args.restarter)
    script_path = os.path.join(script_dir, label)
    print(script_path)
    print(restarter_path)
    cmd = 'python {} --script {} --max-epoch {} --interval {}'.format(restarter_path, script_path, args.max_epoch, args.interval)
    print(cmd)
    outfile = args.script + '.rs.sh'
    with open(outfile, 'w', encoding='utf8') as fout:
        fout.write(cmd)
        fout.write('\n')
        fout.close()

    new_script_dir = script_dir.replace('/hdfs', 'gfs://%s'%args.vc)
    cmd = 'philly-script.py {} {}'.format(args.script, new_script_dir)
    print(cmd)
    os.system(cmd)

    cmd = 'philly-script.py {} {}'.format(outfile, new_script_dir)
    print(cmd)
    os.system(cmd)





