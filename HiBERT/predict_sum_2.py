import collections
import itertools
import os, sys
import math
import torch
import numpy
import argparse # For potential simpler epoch_itr, though not strictly used in this version

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

    # These CUDA checks are commented out for CPU execution
    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    utils.xprintln('setup task done!')

    # Load dataset splits
    # Consider if train split is needed if only doing prediction and it's large
    load_dataset_splits(args, task, ['train'])
    load_dataset_splits(args, task, ['valid', 'test'], shuffle=False)
    utils.xprintln('load dataset done!')

    if args.task == 'extractive_summarization':
        if distributed_utils.is_master(args): # Should be true for single CPU process
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
    # print(model) # Can be very verbose, uncomment if needed
    sys.stdout.flush()

    # if summarization try to load pretrained model
    if args.task == 'extractive_summarization' or args.task == 'pretrain_document_modeling':
        if args.pretrained_sum_model:
            # Load directly to CPU
            state = torch.load(args.pretrained_sum_model, map_location='cpu', weights_only=False)
            params = state['model']
            # print( params.keys() ) # Can be verbose, uncomment if needed
            model.load_state_dict(params, strict=False)
            print('load pretrained sum model from {} done!'.format(args.pretrained_sum_model))

    # Build trainer
    if args.fp16:
        if torch.cuda.is_available():
            trainer = FP16Trainer(args, task, model, criterion)
        else:
            print('| WARNING: FP16 training is requested but CUDA is not available. Disabling FP16 and using standard Trainer.')
            args.fp16 = False # Ensure fp16 is off for CPU
            trainer = Trainer(args, task, model, criterion)
    else:
        # Standard Trainer, can run on CPU or GPU
        if torch.cuda.is_available(): # Only check capability if CUDA is available
            try:
                # Use args.device_id if available, otherwise default to 0 for the check
                device_to_check = args.device_id if hasattr(args, 'device_id') and args.device_id is not None else 0
                if torch.cuda.get_device_capability(device_to_check)[0] >= 7:
                    print('| NOTICE: your CUDA device may support faster training with --fp16')
            except Exception as e:
                # Handles cases like invalid device_id or other CUDA issues
                print(f"| INFO: Could not get CUDA device capability: {e}")
        trainer = Trainer(args, task, model, criterion)
    
    # Print device information
    if torch.cuda.is_available() and args.distributed_world_size > 0 :
        print(f'| Predicting on {args.distributed_world_size} GPUs')
        print(f'| max tokens per GPU = {args.max_tokens}, max sentences per GPU = {args.max_sentences}')
    else:
        print('| Predicting on CPU')
        # For CPU, distributed_world_size should be 1 if running in the 'else' main branch
        # and args.distributed_rank should be 0
        effective_world_size = 1 # For single CPU process
        print(f'| max tokens = {args.max_tokens}, max sentences = {args.max_sentences}')


    # Initialize dataloader
    # The main training loop is commented out.
    # epoch_itr is passed to validate and validate_metric, mainly for epoch_itr.epoch.
    max_positions = trainer.get_model().max_positions()
    epoch_itr = data.EpochBatchIterator(
        dataset=task.dataset(args.train_subset), # This might be slow if train_subset is large
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=1, # 1 is fine for CPU
        seed=args.seed,
        num_shards=args.distributed_world_size if torch.cuda.is_available() and args.distributed_world_size > 0 else 1,
        shard_id=args.distributed_rank if torch.cuda.is_available() and args.distributed_world_size > 0 else 0,
    )
    # Alternative for simpler/faster init if only .epoch is needed by validation functions:
    # epoch_itr = argparse.Namespace(epoch=1) # Requires import argparse

    # will not do any training (original comment)
    '''
    # Load the latest checkpoint if one is available
    load_checkpoint(args, trainer, epoch_itr)
    # ...rest of the commented training setup...
    '''

    # Train until the learning rate gets too small (original comment, but loop is commented)
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')

    # Directly call validation if only predicting
    if args.task == 'extractive_summarization':
        if distributed_utils.is_master(args): # True for single CPU process
            
            subsets_to_evaluate_metrics = []
            # Add 'valid' subsets if raw_valid is provided
            if args.raw_valid and args.valid_subset:
                subsets_to_evaluate_metrics.extend(args.valid_subset.split(','))
            
            # Add 'test' subset if raw_test is provided
            if args.raw_test:
                # Assuming the standard name for the test split is 'test'
                # and it should be loaded by task.dataset('test')
                if 'test' not in subsets_to_evaluate_metrics: # Avoid duplicates
                    subsets_to_evaluate_metrics.append('test')

            if not subsets_to_evaluate_metrics:
                print("| No subsets specified for ROUGE metric evaluation (e.g., via --raw-valid or --raw-test). Skipping ROUGE summarization.")
            else:
                # Generate model predictions (.txt files) for all specified subsets
                validate_metric(args, trainer, task, epoch_itr, subsets_to_evaluate_metrics)
                
                # Add ROUGE evaluation jobs for each subset
                for subset_name in subsets_to_evaluate_metrics:
                    if subset_name not in sum_pool_params:
                        print(f"| WARNING: Subset '{subset_name}' not found in sum_pool_params. Skipping ROUGE for it.")
                        continue
                    
                    # Path to the model's prediction output file (generated by validate_metric)
                    model_output_txt_file = os.path.join(args.save_dir, '{}.{}.txt'.format(epoch_itr.epoch, subset_name))
                    # Path for the ROUGE script's output
                    rouge_output_file = os.path.join(args.save_dir, '{}.{}'.format(epoch_itr.epoch, subset_name))
                    
                    print(f'| Adding ROUGE eval job for subset: {subset_name}')
                    print(f'|   Model output: {model_output_txt_file}')
                    print(f'|   ROUGE output: {rouge_output_file}')

                    # Ensure the .txt file from validate_metric exists before adding job
                    if not os.path.exists(model_output_txt_file):
                        print(f"| WARNING: Model output file {model_output_txt_file} not found. Skipping ROUGE for {subset_name}.")
                        continue

                    current_sum_params = sum_pool_params[subset_name]
                    
                    # Add job for ROUGE (without reranking)
                    sum_eval_pool.add_eval_job(**make_params(current_sum_params, model_output_txt_file, rouge_output_file, False, False))
                    # Add job for ROUGE (with reranking, if applicable, though here it's False)
                    # This matches the original logic of adding two jobs.
                    sum_eval_pool.add_eval_job(**make_params(current_sum_params, model_output_txt_file, rouge_output_file, True, False))
    
    # The validate function is also called for metrics like loss/perplexity
    # This typically uses args.valid_subset
    valid_losses = validate(args, trainer, task, epoch_itr, args.valid_subset.split(','))

    '''
    # Main training loop is commented out
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # ...
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))
    '''

    if args.task == 'extractive_summarization':
        if distributed_utils.is_master(args): # True for single CPU process
            if not subsets_to_evaluate_metrics: # Check if ROUGE eval was skipped
                 print("| ROUGE summarization skipped as no subsets were evaluated for metrics.")
            else:
                sum_eval_pool.join()
                from sum_eval import summarize_rouge
                summarize_rouge(args.save_dir)


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # This function is not called if the main training loop is commented out.
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
            required_batch_size_multiple=1, # 1 is fine for CPU
            seed=args.seed,
            num_shards=args.distributed_world_size if torch.cuda.is_available() and args.distributed_world_size > 0 else 1,
            shard_id=args.distributed_rank if torch.cuda.is_available() and args.distributed_world_size > 0 else 0,
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
            if sample is None:
                continue

            # Debug shapes before forward pass
            print("Pre-forward shapes:")
            print("- Input shape:", sample['net_input']['src_tokens'].shape)
            print("- Target shape:", sample['target'].shape if 'target' in sample else "No target")

            # Forward pass
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
    if not distributed_utils.is_master(args):
        return

    # Get the device the model is on
    device = next(trainer.model.parameters()).device

    for subset in subsets:
        model_output_file = os.path.join(args.save_dir, '{}.{}.txt'.format(epoch_itr.epoch, subset))
        fout = open(model_output_file, 'w', encoding='utf8')
        
        # Write dictionary info first
        fout.write('%d\n'%len(task.target_dictionary))
        for i in range(len(task.target_dictionary)):
            fout.write('{}\t{}\n'.format(task.target_dictionary[i], i))

        itr = data.EpochBatchIterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=trainer.get_model().max_positions(),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=1,
            seed=args.seed,
            num_shards=1,
            shard_id=0,
        ).next_epoch_itr(shuffle=False)
        
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # Variables to collect labels for current document
        current_doc_labels = []
        current_doc_preds = []
        current_doc_dists = []
        current_doc_id = None

        for sample in progress:
            if sample is None:
                continue
                
            # Move all input tensors to the model's device
            sample = utils.move_to_cuda(sample) if torch.cuda.is_available() else utils.move_to_cpu(sample)
                
            trainer.model.eval()
            with torch.no_grad():
                net_output = trainer.model(**sample['net_input'])
                probs = trainer.model.get_normalized_probs(net_output, log_probs=False)
                probs = probs.squeeze(1)  # Remove middle dimension if present
                target = trainer.model.get_targets(sample, net_output)
                if target.dim() > 1:
                    target = target[:, 0]  # Take first position from each sequence
                _, pred = probs.max(1)
                
                # Move tensors back to CPU for numpy operations
                probs = probs.cpu()
                pred = pred.cpu()
                target = target.cpu()
                
                # Process each item in the batch
                for i in range(pred.size(0)):
                    doc_id = sample['id'][i].item() if 'id' in sample else i
                    
                    # If this is a new document, write the previous one
                    if current_doc_id is not None and doc_id != current_doc_id:
                        # Write accumulated document data
                        fout.write('True      Labels:\t{}\n'.format(' '.join(current_doc_labels)))
                        fout.write('Predicted Labels:\t{}\n'.format(' '.join(current_doc_preds)))
                        fout.write('Predicted Distri:\t{}\n'.format(' | '.join(current_doc_dists)))
                        fout.flush()
                        # Reset for new document
                        current_doc_labels = []
                        current_doc_preds = []
                        current_doc_dists = []
                    
                    # Store current sentence info
                    current_doc_id = doc_id
                    current_doc_labels.append(task.target_dictionary[target[i]])
                    current_doc_preds.append(task.target_dictionary[pred[i]])
                    current_doc_dists.append(' '.join(map(lambda x: str(x), probs[i].tolist())))

        # Write the last document if there's data
        if current_doc_labels:
            fout.write('True      Labels:\t{}\n'.format(' '.join(current_doc_labels)))
            fout.write('Predicted Labels:\t{}\n'.format(' '.join(current_doc_preds)))
            fout.write('Predicted Distri:\t{}\n'.format(' | '.join(current_doc_dists)))

        fout.close()
        utils.xprintln('valid metric %s done!' % model_output_file)


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
    if hasattr(save_checkpoint, 'best'): # save_checkpoint might not be defined if training loop is skipped
        if hasattr(save_checkpoint, 'best') and save_checkpoint.best is not None:
             stats['best'] = min(save_checkpoint.best, stats['valid_loss'])
        else:
            stats['best'] = stats['valid_loss'] # Initialize best if not present
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')
    except TypeError: # Handle cases where loss might be None
        return float('inf')


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    # This function is not called if the main training loop is commented out.
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
    else: # If val_loss is None, keep previous best or initialize if not set
        save_checkpoint.best = prev_best if hasattr(save_checkpoint, 'best') else float('inf')


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
            if os.path.exists(old_chk): # Check if file exists before removing
                os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    # This function is not called if the main training loop is commented out.
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path) # map_location handled by Trainer
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
                task.load_dataset(split_k, shuffle=shuffle) # Pass shuffle argument
                print('| {} {} {} examples'.format(args.data if args.data else "Dataset", split_k, len(task.dataset(split_k))))
            except FileNotFoundError as e:
                if k > 0:
                    break
                raise e
            except Exception as e: # Catch other potential errors during dataset loading
                print(f"| Error loading dataset split {split_k}: {e}")
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
        # This path is for distributed GPU training.
        # For CPU, we expect to fall into the 'else' block.
        from distributed_train import main as distributed_main
        distributed_main(args)
    elif args.distributed_world_size > 1:
        # This path is for multiprocessing (potentially multiple GPUs or CPU cores if supported by fairseq's setup)
        # For single CPU, we expect to fall into the 'else' block.
        from multiprocessing_train import main as multiprocessing_main
        multiprocessing_main(args)
    else:
        # This is the path for single process execution (expected for single CPU)
        # Ensure distributed_rank and distributed_world_size are appropriate for single CPU
        if not hasattr(args, 'distributed_rank'):
            args.distributed_rank = 0
        if not hasattr(args, 'distributed_world_size') or args.distributed_world_size is None : # Check for None as well
            args.distributed_world_size = 1
        
        # If CUDA is not available, force distributed world size to 1 and rank to 0
        # This helps ensure fairseq utilities behave correctly in a non-distributed CPU context
        if not torch.cuda.is_available():
            args.distributed_world_size = 1
            args.distributed_rank = 0
            if hasattr(args, 'device_id'): # device_id is CUDA specific
                args.device_id = None # or None, indicate no GPU

        main(args)