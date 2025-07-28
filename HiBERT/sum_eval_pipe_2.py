import argparse
from sum_eval import MultiProcSumEval, summarize_rouge, evaluate_extractive
import os, re, sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ncpu', type=int, default=1)
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
                              entity_map_file=None,
                              length=-1, eval_type='predict',
                              topk=opts.topk, rerank=False, with_m=False,
                              cmd='-a -c 95 -m -n 4 -w 1.2',
                              add_full_stop=opts.add_full_stop,
                              trigram_block=not opts.no_trigram_block)
    
    # Get all model checkpoints
    model_files = [f for f in os.listdir(opts.model_dir) if f.endswith('.result')]
    model_files.sort()  # Sort to process in order

    # Evaluate each model checkpoint
    for model_file in model_files:
        model_path = os.path.join(opts.model_dir, model_file)
        out_rouge_file = model_path + '.valid.rouge'
        
        # Evaluate on validation set
        eval_pool.add_eval_job(
            article_file=valid_pool_params['article_file'],
            summary_file=valid_pool_params['summary_file'],
            entity_map_file=valid_pool_params['entity_map_file'],
            result_file=model_path,
            out_rouge_file=out_rouge_file,
            length=valid_pool_params['length'],
            eval_type=valid_pool_params['eval_type'],
            topk=valid_pool_params['topk'],
            rerank=valid_pool_params['rerank'],
            with_m=valid_pool_params['with_m'],
            cmd=valid_pool_params['cmd'],
            add_full_stop=valid_pool_params['add_full_stop'],
            trigram_block=valid_pool_params['trigram_block']
        )

        # Evaluate on test set if specified
        if opts.raw_test:
            test_rouge_file = model_path + '.test.rouge'
            eval_pool.add_eval_job(
                article_file=opts.raw_test + '.article',
                summary_file=opts.raw_test + '.summary',
                entity_map_file=None,
                result_file=model_path,
                out_rouge_file=test_rouge_file,
                length=-1,
                eval_type='predict',
                topk=opts.topk,
                rerank=False,
                with_m=False,
                cmd='-a -c 95 -m -n 4 -w 1.2',
                add_full_stop=opts.add_full_stop,
                trigram_block=not opts.no_trigram_block
            )

    eval_pool.join()
    print('evaluation done!')
