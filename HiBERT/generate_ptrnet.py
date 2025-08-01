
#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from fairseq import bleu, data, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.pointer_generator import PointerGenerator
from fairseq.sequence_scorer import SequenceScorer


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    utils.xpprint(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task)

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(beamable_mm_beam_size=None if args.no_beamable_mm else args.beam)
        if args.fp16:
            model.half()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = data.EpochBatchIterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=models[0].max_positions(),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    if args.score_reference:
        # translator = SequenceScorer(models, task.target_dictionary)
        raise Exception('score_reference not supported yet!')
    else:
        translator = PointerGenerator(
            models, task.source_dictionary, task.target_dictionary, beam_size=args.beam,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
            dataset=task.dataset(args.gen_subset) if args.raw_text else None,
        )

    if use_cuda:
        translator.cuda()

    # fout_ptr = open(args.predict, 'w', encoding='utf8')
    predict_pointers = {}
    fullstop_counts = 0

    # Generate and compute BLEU score
    scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    with progress_bar.build_progress_bar(args, itr) as t:
        if args.score_reference:
            translations = translator.score_batched_itr(t, cuda=use_cuda, timer=gen_timer)
        else:
            translations = translator.generate_batched_itr(
                t, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
                cuda=use_cuda, timer=gen_timer, prefix_size=args.prefix_size,
            )

        wps_meter = TimeMeter()
        for sample_id, src_tokens, target_tokens, target_ptrs, hypos in translations:
            # print( target_ptrs )
            # Process input and ground truth
            has_target = target_tokens is not None
            target_tokens = target_tokens.int().cpu() if has_target else None

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
            else:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                if has_target:
                    target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

            if not args.quiet:
                print('S-{}\t{}'.format(sample_id, src_str))
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str))

            # Process top predictions
            for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu(),
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    attention=hypo['attention'].cpu(),
                )

                # produce predictions
                assert hypo['tokens'].size(0) == hypo['pointers'].size(0)
                lbls = ['F'] * src_tokens.size(0)
                for p in hypo['pointers']:
                    lbls[p.item()] = 'T'
                if src_tokens[-2].item() == src_dict.index('.'):
                    lbls[-2] = 'T'
                    fullstop_counts += 1
                predict_pointers[sample_id.item()] = ( lbls[0:-1], hypo['pointers'] )

                '''
                s_tokens = src_tokens[ src_tokens.ne( src_dict.pad() ) ]
                print( "***", s_tokens.size(), "***" )
                print( src_tokens )
                '''

                if not args.quiet:
                    print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                    print('P-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            hypo['positional_scores'].tolist(),
                        ))
                    ))
                    print('A-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(lambda x: str(utils.item(x)), alignment))
                    ))

                # Score only the top hypothesis
                if has_target and i == 0:
                    if align_dict is not None or args.remove_bpe is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tokenizer.Tokenizer.tokenize(
                            target_str, tgt_dict, add_if_not_exist=True)
                    scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(src_tokens.size(0))
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += 1

            # this is for test
            '''
            if num_sentences >= 5:
                break
            '''

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))

    print( 'fullstop_counts = ', fullstop_counts )
    with open(args.predict_file, 'w', encoding='utf8') as fout_ptr:
        items = list( predict_pointers.items() )
        items.sort( key=lambda x: x[0] )
        for _, item in items:
            lbl = item[0]
            fout_ptr.write( ' '.join(lbl) + '\n' )

    from evaluation.prec_recall_f1 import get_prec_recall_f1
    metrics = get_prec_recall_f1(args.predict_file, args.gold_file, show=True)
    print( metrics )


if __name__ == '__main__':
    parser = options.get_generation_parser()
    parser.add_argument('--predict-file', default='predict.tf.lbl',
                        help='prediction file')
    parser.add_argument('--gold-file', default='/home/xizhang/Projects/extractive_compress/dataset/google_200k/test.lbl',
                        help='prediction file')
    args = options.parse_args_and_arch(parser)
    main(args)
