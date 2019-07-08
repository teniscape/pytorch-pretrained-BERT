# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OpenAI GPT model fine-tuning script."""
import argparse
import os
import csv
import random
import json
import logging
import pandas as pd
import shutil
import time
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert import (OpenAIGPTDoubleHeadsModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                     OpenAIAdam, WEIGHTS_NAME, CONFIG_NAME,
                                     GPT2DoubleHeadsModel, GPT2LMHeadModel, GPT2Tokenizer)
from pytorch_pretrained_bert.optimization import WarmupLinearSchedule

DATA_DIR = '../data'.format(os.getenv('HOME'))

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def load_dataset(split, task_name, debug=False, seed=42):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    assert split in {'train', 'dev'}, 'Split "{}" not yet supported'.format(split)
    if task_name == 'rocstories':
        file_version = 'test' if split == 'dev' else 'val'
        dataset_path = '{0}/rocstories/cloze_test_{1}__spring2016 - cloze_test_ALL_{1}.csv'.format(
            DATA_DIR, file_version)
        with open(dataset_path, encoding='utf_8') as f:
            f = csv.reader(f)
            output = []
            next(f)  # Skip the first line
            for line in tqdm(f):
                output.append((' '.join(line[1:5]), line[5], line[6], int(line[-1])-1))
    elif task_name == 'sqa-lm':
        sqa_split = 'test' if split == 'dev' else 'train'
        wtq_split = 'pristine-unseen-tables' if split == 'test' else 'training'

        df_subq = pd.read_csv('{}/sqa/{}.tsv'.format(DATA_DIR, sqa_split),
                              delimiter='\t', encoding='utf-8')
        df_q = pd.read_csv('{}/WikiTableQuestions/data/{}.tsv'.format(DATA_DIR, wtq_split),
                           delimiter='\t', encoding='utf-8')

        qids_with_subqs = list(set(df_subq['id']))
        qids_with_subqs.sort()

        examples = []
        for qid in tqdm(qids_with_subqs):
            qid = qid.replace('ns', 'nt')
            assert len(df_q[df_q.id == qid].utterance.values) > 0, 'Invalid QID: {}'.format(qid)
            q = df_q[df_q.id == qid].utterance.values[0]
            df_subq_qid = df_subq[df_subq.id == qid]
            annotators = list(set(df_subq_qid.annotator))
            annotators.sort()
            for annotator in annotators:
                df_subq_qid_annotator = df_subq_qid[df_subq_qid.annotator == annotator]
                positions = list(set(df_subq_qid_annotator.position))
                positions.sort()
                subqs = [df_subq_qid_annotator[df_subq_qid_annotator.position == position].question.values[0].strip()
                         for position in positions]
                example = ' '.join([q] + subqs).strip()
                examples.append(example)

        print('Processed {} examples.'.format(len(examples)))
        return examples
    else:
        file_path = '{}/squad/{}-v2.0.json'.format(DATA_DIR, split)
        with open(file_path, 'r') as f:
            data = json.load(f)

        output = []
        shuffler = random.Random(seed)
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        for article in tqdm(data['data']):
            for paragraph in article['paragraphs']:
                qs = [tokenizer.decode((tokenizer.encode(qa['question'].strip()))).strip() for qa in paragraph['qas']]
                if task_name == 'squad-questions-lm':
                    output += qs
                else:  # squad-questions-cond-lm
                    shuffler.shuffle(qs)
                    if (len(qs) % 2) == 1:
                        # qs.append('')  # Use for generative model. Doesn't encourage repeating the original Q.
                        qs.append(qs[-1])  # Use for ranking model. Pair last Q with itself if it would be unpaired.
                    for q1, q2 in zip(qs[::2], qs[1::2]):
                        output.append((q1 + ' ' + q2).strip())
            if debug and (len(output) > 100):
                break
    return output


def pre_process_datasets(encoded_datasets, input_len, cap_length, task_name,
                         start_token=None, delimiter_token=None, clf_token=None):
    """ Pre-process datasets containing lists of tuples:
        - [ROCStories] (story, 1st continuation, 2nd continuation, label)
        - [Language Modeling] (sequence of tokens)
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        if task_name == 'rocstories':
            input_ids = np.zeros((n_batch, 2, input_len), dtype=np.int64)
            lm_labels = np.full((n_batch, 2, input_len), fill_value=-1, dtype=np.int64)
            mc_token_ids = np.zeros((n_batch, 2), dtype=np.int64)
            mc_labels = np.zeros((n_batch,), dtype=np.int64)
            for i, (story, cont1, cont2, mc_label), in enumerate(dataset):
                with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
                with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
                input_ids[i, 0, :len(with_cont1)] = with_cont1
                input_ids[i, 1, :len(with_cont2)] = with_cont2
                mc_token_ids[i, 0] = len(with_cont1) - 1
                mc_token_ids[i, 1] = len(with_cont2) - 1
                lm_labels[i, 0, :len(with_cont1)] = with_cont1
                lm_labels[i, 1, :len(with_cont2)] = with_cont2
                mc_labels[i] = mc_label
            all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
        else:
            input_ids = np.zeros((n_batch, 1, input_len), dtype=np.int64)
            lm_labels = np.full((n_batch, 1, input_len), fill_value=-1, dtype=np.int64)
            for i, seq, in enumerate(dataset):
                input_ids[i, 0, :len(seq)] = seq
                lm_labels[i, 0, :len(seq)] = seq
            all_inputs = (input_ids, lm_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets


def get_tokenizer_class(model_name):
    """ Returns a tokenizer's Python class """
    return OpenAIGPTTokenizer if model_name == 'openai-gpt' else GPT2Tokenizer


def get_model_class(model_name, task_name):
    """ Returns a model's Python class """
    if task_name == 'rocstories':
        return OpenAIGPTDoubleHeadsModel if model_name == 'openai-gpt' else GPT2DoubleHeadsModel
    else:
        return OpenAIGPTLMHeadModel if model_name == 'openai-gpt' else GPT2LMHeadModel


def load_model(saved_dir):
    """ Loads a previously saved model """
    output_args_file = os.path.join(saved_dir, 'training_args.bin')
    args = torch.load(output_args_file)
    print('Loaded args:', args)
    tokenizer_class = get_tokenizer_class(args.model_name)
    tokenizer = tokenizer_class.from_pretrained(saved_dir)
    model_class = get_model_class(args.model_name, args.task_name)
    model = model_class.from_pretrained(saved_dir)
    return model, tokenizer, args


def save_model(model, tokenizer, args, output_dir, weights_name=WEIGHTS_NAME, override_default_weights=True):
    """ Saves and existing model """
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, weights_name)
    output_default_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    if override_default_weights:
        torch.save(model_to_save.state_dict(), output_default_model_file)

    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

    output_args_file = os.path.join(output_dir, 'training_args.bin')
    torch.save(args, output_args_file)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='openai-gpt',  # {'openai-gpt', 'gpt2'}
                        help='pretrained model name')
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=32)  # {16, 32, 64} -> {128, 256}
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)  # {6.25e-5, 1.25e-4, 2.5e-4} -> {3.125e-5, ...}
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--debug', action='store_true', help="Whether to use debug mode")
    # TODO: local_rank, gradient_accumulation_steps

    args = parser.parse_args()
    args.task_name = args.task_name.lower()
    print(args)
    assert args.task_name in {'rocstories', 'squad-questions-lm', 'squad-questions-cond-lm', 'sqa-lm'}, \
        'Unimplemented task_name {}'.format(args.task_name)

    eval_batch_size = 2 * args.train_batch_size
    output_dir = 'checkpoint/tn={}.mn={}.tbs={}.lr={}'.format(
        args.task_name, args.model_name, args.train_batch_size, args.learning_rate)
    print('Saving to {}'.format(output_dir))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}, 16-bits training: {}".format(device, n_gpu, args.fp16))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif args.overwrite_output_dir:
        print('Overwriting existing output directory', output_dir)
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    special_tokens = ['_start_', '_delimiter_', '_classify_'] if args.task_name == 'rocstories' else None
    tokenizer_class = get_tokenizer_class(args.model_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_name, special_tokens=special_tokens)
    model_class = get_model_class(args.model_name, args.task_name)
    model = model_class.from_pretrained(args.model_name,
                                        num_special_tokens=len(special_tokens) if special_tokens else 0)
    if args.fp16:
        model.half()
    model.to(device)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens) \
        if special_tokens else []

    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        return list(tokenize_and_encode(o) for o in obj)

    logger.info("Encoding dataset...")
    train_dataset = load_dataset('train', args.task_name, args.debug, args.seed)
    eval_dataset = load_dataset('dev', args.task_name, args.debug, args.seed)
    datasets = (train_dataset, eval_dataset)
    encoded_datasets = tokenize_and_encode(datasets)

    # Compute the max input length for the Transformer
    max_length = model.config.n_positions // 2 - 2
    if args.task_name == 'rocstories':
        input_length = max(len(story[:max_length]) + max(len(cont1[:max_length]), len(cont2[:max_length])) + 3
                               for dataset in encoded_datasets for story, cont1, cont2, _ in dataset)
    else:
        input_length = max(len(seq) for dataset in encoded_datasets for seq in dataset)
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model
    print('input_length =', input_length)

    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_length, args.task_name,
                                           *special_tokens_ids)
    train_tensor_dataset, eval_tensor_dataset = tensor_datasets[0], tensor_datasets[1]

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        optimizer = OpenAIAdam(optimizer_grouped_parameters,
                               lr=args.learning_rate,
                               warmup=args.warmup_proportion,
                               max_grad_norm=args.max_grad_norm,
                               weight_decay=args.weight_decay,
                               t_total=num_train_optimization_steps)

    # Train loop
    tb_writer = SummaryWriter(output_dir)
    global_step, nb_tr_steps, tr_loss, exp_average_loss, best_eval_loss = 0, 0, 0, None, float('inf')
    patience_left = args.patience
    start_time = time.time()
    for epoch_no in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        nb_tr_steps, tr_loss = 0, 0
        tqdm_bar = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(tqdm_bar):
            batch = tuple(t.to(device) for t in batch)
            if args.task_name == 'rocstories':
                input_ids, mc_token_ids, lm_labels, mc_labels = batch
                losses = model(input_ids, mc_token_ids, lm_labels, mc_labels)
                loss = args.lm_coef * losses[0] + losses[1]
            else:
                input_ids, lm_labels = batch
                loss = model(input_ids, lm_labels=lm_labels)

            if args.fp16:
                optimizer.backward(loss)
                # modify learning rate with special warm up used
                # if args.fp16 is False, the default optimizer is used that handles this automatically
                lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            else:
                loss.backward()
                lr_this_step = optimizer.get_lr()[0]
            optimizer.step()
            optimizer.zero_grad()
            tr_loss += loss.item()
            exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
            global_step += 1
            nb_tr_steps += 1
            tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, lr_this_step)
            tb_writer.add_scalar('lr', lr_this_step, global_step * args.train_batch_size)
            tb_writer.add_scalar('loss', loss.item(), global_step * args.train_batch_size)
            tb_writer.add_scalar('exp_average_loss', exp_average_loss, global_step * args.train_batch_size)

        # Validation
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            if args.task_name == 'rocstories':
                input_ids, mc_token_ids, lm_labels, mc_labels = batch
                with torch.no_grad():
                    lm_loss, mc_loss = model(input_ids, mc_token_ids, lm_labels, mc_labels)
                    loss = args.lm_coef * lm_loss + mc_loss
                    mc_logits = model(input_ids, mc_token_ids)[1]

                mc_logits = mc_logits.detach().cpu().numpy()
                mc_labels = mc_labels.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(mc_logits, mc_labels)

                eval_loss += loss.mean().item()
                eval_accuracy += tmp_eval_accuracy
            else:
                input_ids, lm_labels = batch
                with torch.no_grad():
                    lm_loss = model(input_ids, lm_labels=lm_labels)

                eval_loss += lm_loss.mean().item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        tb_writer.add_scalar('eval_loss', eval_loss, global_step * args.train_batch_size)
        result = {'eval_loss': eval_loss,
                  'train_loss': tr_loss / nb_tr_steps}
        if args.task_name == 'rocstories':
            result['eval_accuracy'] = eval_accuracy / nb_eval_examples

        output_eval_file = os.path.join(output_dir, "eval_results_{}.txt".format(epoch_no))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # Model saving and early stopping
        print('Epoch {} complete!'.format(epoch_no))
        if eval_loss < best_eval_loss:
            print('Best loss so far! {} -> {}'.format(best_eval_loss, eval_loss))
            best_eval_loss = eval_loss
            save_model(model, tokenizer, args, output_dir, 'model_epoch_{}.bin'.format(epoch_no), True)
            patience_left = args.patience
        else:
            print('Loss up from best epoch: {} -> {}'.format(best_eval_loss, eval_loss))
            save_model(model, tokenizer, args, output_dir, 'model_epoch_{}.bin'.format(epoch_no), False)
            patience_left -= 1
            if patience_left <= 0:
                print('Ran out of patience. Stopping training.')
                break

    print('Completed training in {}s!'.format(time.time() - start_time))


if __name__ == '__main__':
    main()

