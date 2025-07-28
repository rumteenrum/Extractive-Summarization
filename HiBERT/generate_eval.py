
import argparse, os, sys
from bleu_eval import bleu_eval
from pyrougex import get_rouge_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', default='/home/xizhang/Projects/abs_sent_compress/experiments/abs_sent_compress_small/model_train.transformer.small.cnn_dm.philly.u8.4L.lr1.5.2.sh/gen_test_dm/test_dm.sh.43.new.unk.txt')
    parser.add_argument('--datadir', default='/home/xizhang/Projects/seq2seq/dataset/cnn_dailymail_lei+deepmind/fairseq_format_no_bpe/cnn_dm_bin_50k')
    parser.add_argument('--src-tag', default='.src')
    parser.add_argument('--tgt-tag', default='.tgt')
    parser.add_argument('--model', default='/home/xizhang/Projects/abs_sent_compress/experiments/abs_sent_compress_small/model_train.transformer.small.cnn_dm.philly.u8.4L.lr1.5.2.sh/checkpoint43.pt')
    parser.add_argument('--gen-subset', default='test_dm')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--replace-unk', action='store_true', help='replace unk is always on!')

    args = parser.parse_args()
    args.__dict__['log_file'] = args.out_file + '.log.txt'
    args.__dict__['src_file'] = os.path.join(args.datadir, args.gen_subset + args.src_tag)
    args.__dict__['tgt_file'] = os.path.join(args.datadir, args.gen_subset + args.tgt_tag)

    if not os.path.exists(args.src_file):
        print('%s does not exist!'%args.src_file)
        print(os.listdir(args.datadir))
        sys.exit(1)

    return args


def get_generate_fname():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, 'generate.py')


def run_gen_cmd(args):
    cmd = 'python {gen_py} {datadir} --path {model} --raw-text --gen-subset {subset} --batch-size {batchsize} --beam {beam} --replace-unk | tee {log}'.format(
            gen_py=get_generate_fname(),
            datadir=args.datadir,
            model=args.model,
            subset=args.gen_subset,
            batchsize=args.batch_size,
            beam=args.beam,
            log=args.log_file
    )
    print(cmd)
    os.system(cmd)


def analyze(src_file, ref_file, sys_file, out_file, encoding='utf8'):
    fout = open(out_file, 'w', encoding=encoding)
    fin_src = open(src_file, encoding=encoding)
    fin_ref = open(ref_file, encoding=encoding)
    fin_sys = open(sys_file)
    while True:
        src = fin_src.readline()
        ref = fin_ref.readline()
        sys = fin_sys.readline()

        if (not src) or (not ref) or (not sys): break
        fout.write('[source] = %s'%src)
        fout.write('[refere] = %s' % ref)
        fout.write('[output] = %s' % sys)
        fout.write('\n')

    fin_src.close()
    fin_ref.close()
    fin_sys.close()
    fout.close()


def get_outfile(log_file, out_file):
    lines = []
    for line in open(log_file, encoding='utf8'):
        if not line.startswith('H-'): continue
        # print(line.strip())
        fds = line.strip().split('\t')
        assert len(fds) == 3
        lid = int(fds[0][2:])
        lines.append( (lid, fds[2]) )

    lines.sort(key = lambda x: x[0])
    with open(out_file, 'w', encoding='utf8') as fout:
        for item in lines:
            fout.write(item[1])
            fout.write('\n')


def eval_log(log_file, out_file, tgt_file):
    get_outfile(log_file, out_file)
    bleu_eval(out_file, tgt_file)
    get_rouge_file(out_file, tgt_file, print_rouge=True)


if __name__ == '__main__':
    args = get_args()
    run_gen_cmd(args)
    eval_log(args.log_file, args.out_file, args.tgt_file)
    analyze(args.src_file, args.tgt_file, args.out_file, args.out_file + '.ana.txt')
