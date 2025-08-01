
import collections
import itertools
import os, sys
import math
import torch
import numpy

from fairseq import data, distributed_utils, options, progress_bar, tasks, utils
from fairseq.fp16_trainer import FP16Trainer
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter


def main(args):
    # we should not do this!
    '''
    if args.max_tokens is None:
        args.max_tokens = 6000
    '''
    utils.xpprint(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    utils.xprintln('setup task done!')

    # Load dataset splits
    load_dataset_splits(args, task, ['train'])
    load_dataset_splits(args, task, ['valid', 'test'], shuffle=False)
    utils.xprintln('load dataset done!')

    if args.task == 'extractive_summarization':
        if distributed_utils.is_master(args):
            from sum_eval import MultiProcSumEval
            sum_eval_pool = MultiProcSumEval(args.ncpu_eval)
            sum_valid_pool_params = dict(article_file=args.raw_valid + '.article',
                                          summary_file=args.raw_valid + '.summary',
                                          entity_map_file=None,
                                          length=-1, eval_type='predict',
                                          topk=args.topk_sent_eval, rerank=False, with_m=False,
                                          cmd=args.rouge_155_options, multi_ref=args.multi_ref_sum_eval)

            sum_test_pool_params = dict(article_file=args.raw_test + '.article',
                                          summary_file=args.raw_test + '.summary',
                                          entity_map_file=None,
                                          length=-1, eval_type='predict',
                                          topk=args.topk_sent_eval, rerank=False, with_m=False,
                                          cmd=args.rouge_155_options, multi_ref=args.multi_ref_sum_eval)
            sum_pool_params = dict(valid=sum_valid_pool_params, test=sum_test_pool_params)

            def make_params(default_dict, result_file, out_rouge_file, rerank=False, with_m=False):
                para_dict = dict(default_dict)
                para_dict['result_file'] = result_file
                para_dict['out_rouge_file'] = out_rouge_file
                para_dict['rerank'] = rerank
                para_dict['with_m'] = with_m
                return para_dict

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))
    print(model)
    import sys
    sys.stdout.flush()

    # if summarization try to load pretrained model
    if args.task == 'extractive_summarization' or args.task == 'pretrain_document_modeling':
        # assume this is a single GPU program
        '''
        if args.init_from_pretrained_doc_model:
            task.load_pretrained_model(model, args.pretrained_doc_model_path)
        '''
        if args.pretrained_sum_model:
            from torch.serialization import default_restore_location
            state = torch.load(args.pretrained_sum_model, map_location=lambda s, l: default_restore_location(s, 'cpu'), weights_only=False)
            params = state['model']
            print( params.keys() )
            model.load_state_dict(params, strict=True)
            print('load pretrained sum model from {} done!'.format(args.pretrained_sum_model))

    # Build trainer
    if args.fp16:
        trainer = FP16Trainer(args, task, model, criterion)
    else:
        if torch.cuda.get_device_capability(0)[0] >= 7:
            print('| NOTICE: your device may support faster training with --fp16')
        trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Initialize dataloader
    max_positions = trainer.get_model().max_positions()
    epoch_itr = data.EpochBatchIterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=1,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )

    # will not do any training
    '''
    # Load the latest checkpoint if one is available
    load_checkpoint(args, trainer, epoch_itr)
    # make sure training from a different checkpoint will use different random seed
    cur_dataset = task.dataset('train')
    if hasattr(cur_dataset, 'rng'):
        print('epoch ', epoch_itr.epoch)
        cur_dataset.rng = numpy.random.RandomState(args.seed+epoch_itr.epoch)

    if hasattr(task, 'run_dummy_batch') and task.run_dummy_batch:
        print('** run dummy batch **')
        # Send a dummy batch to warm the caching allocator
        dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens if args.max_tokens else args.max_sentences, max_positions)
        trainer.dummy_train_step(dummy_batch)
    '''

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')

    if args.task == 'extractive_summarization':
        if distributed_utils.is_master(args):
            validate_metric(args, trainer, task, epoch_itr, valid_subsets)
            for subset in valid_subsets:
                valid_result_file = os.path.join(args.save_dir, '{}.{}.txt'.format(epoch_itr.epoch, subset))
                valid_out_file = os.path.join(args.save_dir, '{}.{}'.format(epoch_itr.epoch, subset))
                print('add job', valid_result_file, valid_out_file)
                sum_eval_pool.add_eval_job(**make_params(sum_pool_params[subset], valid_result_file, valid_out_file, False, False))
                sum_eval_pool.add_eval_job(**make_params(sum_pool_params[subset], valid_result_file, valid_out_file, True, False))
    valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    '''
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if epoch_itr.epoch % args.validate_interval == 0:
            if args.task == 'extractive_summarization':
                if distributed_utils.is_master(args):
                    validate_metric(args, trainer, task, epoch_itr, valid_subsets)
                    for subset in valid_subsets:
                        valid_result_file = os.path.join(args.save_dir, '{}.{}.txt'.format(epoch_itr.epoch, subset))
                        valid_out_file = os.path.join(args.save_dir, '{}.{}'.format(epoch_itr.epoch, subset))
                        sum_eval_pool.add_eval_job(**make_params(sum_pool_params[subset], valid_result_file, valid_out_file, False, False))
                        sum_eval_pool.add_eval_job(**make_params(sum_pool_params[subset], valid_result_file, valid_out_file, True, False))
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))
    '''

    if args.task == 'extractive_summarization':
        if distributed_utils.is_master(args):
            sum_eval_pool.join()
            from sum_eval import summarize_rouge
            summarize_rouge(args.save_dir)


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr()
    progress = progress_bar.build_progress_bar(args, itr, epoch_itr.epoch, no_progress_bar='simple')

    # update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf
    num_batches = len(epoch_itr)
    for i, sample in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        if i < num_batches - 1 and (i + 1) % update_freq > 0:
            # buffer updates according to --update-freq
            trainer.train_step(sample, update_params=False)
            continue
        else:
            log_output = trainer.train_step(sample, update_params=True)

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

        # this is for test
        '''
        if num_updates >= 15:
            break
        '''

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    # reset training meters
    for k in ['train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'clip']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm').avg)
    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = '{:.3f}'.format(trainer.get_meter('loss_scale').avg)
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        cur_dataset = task.dataset(subset)
        if hasattr(cur_dataset, 'rng'):
            # make sure the each valiation use the same random seed
            cur_dataset.rng = numpy.random.RandomState(args.seed)
        # Initialize data iterator
        itr = data.EpochBatchIterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=trainer.get_model().max_positions(),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats)

        valid_losses.append(stats['valid_loss'])
    return valid_losses


def validate_metric(args, trainer, task, epoch_itr, subsets):
    # when training with distributed trainer, only one of them (the one args.distributed_rank == 0) is working ...
    print('args.distributed_rank', args.distributed_rank)
    print('args.distributed_world_size', args.distributed_world_size)
    if not distributed_utils.is_master(args):
        return

    """Evaluate the model on the validation set(s) and return the losses."""
    for subset in subsets:
        model_output_file = os.path.join(args.save_dir, '{}.{}.txt'.format(epoch_itr.epoch, subset))
        fout = open(model_output_file, 'w', encoding='utf8')
        # firstly, output dictionary information
        fout.write('%d\n'%len(task.target_dictionary))
        for i in range(len(task.target_dictionary)):
            fout.write('{}\t{}\n'.format(task.target_dictionary[i], i))
            fout.flush()

        # Initialize data iterator
        itr = data.EpochBatchIterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=trainer.get_model().max_positions(),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=1, # only args.distributed_rank == 0
            shard_id=0, # only args.distributed_rank == 0
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        cnt = 0
        for sample in progress:
            trainer.model.eval()
            sample = utils.move_to_cuda(sample)
            net_output = trainer.model(**sample['net_input'])
            probs = trainer.model.get_normalized_probs(net_output, log_probs=False)
            _, pred = probs.max(2)
            target = trainer.model.get_targets(sample, net_output)
            for i in range(pred.size(0)):
                labels = []
                pred_labels = []
                pred_dists = []
                for j in range(pred.size(1)):
                    if target[i, j] != task.target_dictionary.pad():
                        labels.append( task.target_dictionary[target[i, j]] )
                        pred_labels.append( task.target_dictionary[pred[i, j]] )
                        pred_dists.append( ' '.join( map(lambda x: str(x.item()), probs[i, j]) ) )
                    else:
                        break
                fout.write('True      Labels:\t%s\n'%' '.join(labels))
                fout.write('Predicted Labels:\t%s\n'%' '.join(pred_labels))
                fout.write('Predicted Distri:\t%s\n'%' | '.join(pred_dists))
                fout.flush()
                assert cnt == sample['id'][i]
                cnt += 1
        fout.close()
        utils.xprintln('valid metric %s done!'%model_output_file)


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best'] = min(save_checkpoint.best, stats['valid_loss'])
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
        end_of_epoch and not args.no_epoch_checkpoints and
        epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
        not end_of_epoch and args.save_interval_updates > 0 and
        updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
        val_loss is not None and
        (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'best': save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']


def load_dataset_splits(args, task, splits, shuffle=True):
    for split in splits:
        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            try:
                task.load_dataset(split_k, shuffle)
                print('| {} {} {} examples'.format(args.data, split_k, len(task.dataset(split_k))))
            except FileNotFoundError as e:
                if k > 0:
                    break
                raise e


if __name__ == '__main__':
    parser = options.get_training_parser()
    parser.add_argument('--pretrained-sum-model', help='load pretrained sum model')
    parser.add_argument('--rouge-155-options', default='-a -c 95 -m -n 4 -w 1.2', help='options for ROUGE-155 script')
    parser.add_argument('--multi-ref-sum-eval', action='store_true', help='whether to use multi reference')
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)
