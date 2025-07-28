
import argparse
from sum_eval import MultiProcSumEval, summarize_rouge, evaluate_extractive
import os, re, sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ncpu', type=int, default=4)
    parser.add_argument('-topk', type=int, default=3)
    parser.add_argument('-raw_valid', default='/disk/scratch/XingxingZhang/summarization/dataset/cnn_dailymail_all_in_one/validation')
    parser.add_argument('-raw_test', default='/disk/scratch/XingxingZhang/summarization/dataset/cnn_dailymail_all_in_one/test')
    parser.add_argument('-model_dir', default='/disk/scratch/XingxingZhang/summarization/experiments/extract_baseline/with_eval_multi_ovocab_nopad/run.save.1layer.sh.models')
    parser.add_argument('-add_full_stop', action='store_true')
    parser.add_argument('-no_trigram_block', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    opts = get_args()
    print(opts)
    eval_pool = MultiProcSumEval(opts.ncpu)
    valid_pool_params = dict(article_file=opts.raw_valid + '.article',
                              summary_file=opts.raw_valid + '.summary',
                              entity_map_file=opts.raw_valid + '.entity_map',
                              length=-1, eval_type='predict',
                              topk=opts.topk, rerank=False, with_m=False,
                              cmd='-a -c 95 -m -n 4 -w 1.2',
                              add_full_stop=opts.add_full_stop,
                              trigram_block=not opts.no_trigram_block)

    test_pool_params = dict(article_file=opts.raw_test + '.article',
                              summary_file=opts.raw_test + '.summary',
                              entity_map_file=opts.raw_test + '.entity_map',
                              length=-1, eval_type='predict',
                              topk=opts.topk, rerank=False, with_m=False,
                              cmd='-a -c 95 -m -n 4 -w 1.2',
                              add_full_stop=opts.add_full_stop,
                              trigram_block=not opts.no_trigram_block)

    def make_params(default_dict, result_file, out_rouge_file, rerank=False, with_m=False):
        para_dict = dict(default_dict)
        para_dict['result_file'] = result_file
        para_dict['out_rouge_file'] = out_rouge_file
        para_dict['rerank'] = rerank
        para_dict['with_m'] = with_m
        return para_dict

    def parse_fname(fname):
        m = re.search('(\\d+).(valid).txt', fname)
        if m and len(m.group(0)) == len(fname):
            return int(m.group(1)), None, m.group(2)
        m = re.search('(\\d+)_(\\d+).(valid).txt', fname)
        if m and len(m.group(0)) == len(fname):
            return int(m.group(1)), int(m.group(2)), m.group(3)
        return None, None, None

    model_dir_flist = os.listdir(opts.model_dir)

    def rouge_file_exists(model_dir_flist, epoch, label, rerank):
        prefix = '%s.%s'%(epoch, label)
        for f in model_dir_flist:
            if f.startswith(prefix) and f.endswith('.rouge'):
                if not rerank:
                    return True
                elif 'rerank' in f:
                    return True
        return False


    job_info = []
    for f in os.listdir(opts.model_dir):
        epoch, update, split = parse_fname(f)
        if epoch is not None:
            job_info.append( ('{}'.format(epoch) if update is None else '{}_{}'.format(epoch, update) , opts.model_dir) )
    job_info.sort(key=lambda x: x[0])
    for epoch, model_dir in job_info:
        print(epoch, model_dir)
        valid_result_file = '%s/%s.valid.txt' % (model_dir, epoch)
        valid_out_file = '%s/%s.valid' % (model_dir, epoch)
        if rouge_file_exists(model_dir_flist, epoch, 'valid', False):
            print(valid_out_file, 'exists', False)
            sys.stdout.flush()
        else:
            eval_pool.add_eval_job(**make_params(valid_pool_params, valid_result_file, valid_out_file, False, False))
        if rouge_file_exists(model_dir_flist, epoch, 'valid', True):
            print(valid_out_file, 'exists', True)
            sys.stdout.flush()
        else:
            eval_pool.add_eval_job(**make_params(valid_pool_params, valid_result_file, valid_out_file, True, False))

        test_result_file = '%s/%s.test.txt' % (model_dir, epoch)
        test_out_file = '%s/%s.test' % (model_dir, epoch)
        if rouge_file_exists(model_dir_flist, epoch, 'test', False):
            print(test_out_file, 'exists', False)
            sys.stdout.flush()
        else:
            eval_pool.add_eval_job(**make_params(test_pool_params, test_result_file, test_out_file, False, False))
        if rouge_file_exists(model_dir_flist, epoch, 'test', True):
            print(test_out_file, 'exists', True)
            sys.stdout.flush()
        else:
            eval_pool.add_eval_job(**make_params(test_pool_params, test_result_file, test_out_file, True, False))

    eval_pool.join()
    print('evaluation done!')
    summarize_rouge(opts.model_dir)
