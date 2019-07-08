import argparse
import json
import time
from multiprocessing.pool import ThreadPool
from functools import partial
import numpy as np
import os
import pickle
from pytorch_pretrained_bert import (GPT2Tokenizer, GPT2LMHeadModel, GPT2DoubleHeadsModel,
                                     OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTDoubleHeadsModel)
from tqdm import tqdm
import torch

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

format_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')


def cosine_similarity_chunk(i, X, Y, num_chunks):
    Y_chunked = Y[(i * Y.shape[0]) // num_chunks: ((i + 1) * Y.shape[0]) // num_chunks]
    return cosine_similarity(X, Y_chunked)


def clean_up_tokenization_spaces(out_string):
    """Converts an output string (de-BPE-ed) using de-tokenization algorithm from OpenAI GPT."""
    out_string = out_string.replace('<unk>', '')
    out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ','
            ).replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't"
            ).replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
    return out_string


def word_tokenizer(text):
    return format_tokenizer.decode(format_tokenizer.encode(text), clean_up_tokenization_spaces=False).split()


def word_untokenizer(word_list):
    return clean_up_tokenization_spaces(' '.join(word_list)).strip()


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


def main():
    """
    D_q: Questions from SQuAD + HotpotQA Easy (used to train low-level QA model)
    q_1, ..., q_K: Q's KNNs from D_q, given by word overlap (TF-IDF weighted)
    w_1, ..., w_M: Q's top M TF-IDF weighted tokens
    p_q: Language model trained on D_q

    # Replace <= L words with ones from {w_1, ..., w_M}, while maintaining p_q(q_k') >= p_q(q_k)
    for each q_k \in q_1, ..., q_K:
        Try replacing q_k_i with w_m (M x |q_k| many possible pairs).
        If p_q(q_k'') >= p_q(q_k'), keep this replacement.
        Try this until all w_1, ..., w_M are exhausted.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_name", default="all", type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--top_knn", default=50, type=int, required=False, help="# of top nearest neighbors to modify.")
    parser.add_argument("--num_shards", default=1, type=int, required=False,
                        help="# of total data splits for distributed eval")
    parser.add_argument("--shard_no", default=0, type=int, required=False, help="Distributed eval data split index.")
    args = parser.parse_args()
    args.filter_name = args.filter_name.lower()

    # Load HotpotQA
    qis = []
    cis = []
    idis = []
    for split in ['train', 'dev']:
        with open('data/hotpot-all/{}.json'.format(split), 'r') as f:
            data_hotpot = json.load(f)
        for article in tqdm(data_hotpot['data']):
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    qis.append(qa['question'].strip())
                    cis.append(paragraph['context'])
                    idis.append(qa['id'])
    print('HotpotQA Qs:', len(qis))

    # Load SQuAD 2
    qks = []
    for split in ['train', 'dev']:
        with open('data/squad/{}-v2.0.json'.format(split), 'r') as f:
            data_squad = json.load(f)
        for article in tqdm(data_squad['data']):
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    qks.append(qa['question'].strip())
    print('SQuAD 2 Qs:', len(qks))

    # Fit TFIDF
    start_time = time.time()
    tfidf_filepath = 'data/hotpot-all/tfidf.pkl'
    if os.path.exists(tfidf_filepath):
        tfidf = pickle.load(open(tfidf_filepath, 'rb'))
    else:
        print('TFIDF Fitting...')
        tfidf = TfidfVectorizer(tokenizer=word_tokenizer, min_df=2, analyzer='word')
        tfidf.fit(qis + qks)
        pickle.dump(tfidf, open(tfidf_filepath, 'wb'))
    print('Got TFIDF in {:.0f}s with vocab size {:d}'.format(
        time.time() - start_time, len(tfidf.get_feature_names())))
    
    # HotpotQA: Transform Qs into TFIDF vectors
    start_time = time.time()
    tfidf_hotpot_filepath = 'data/hotpot-all/tfidf_hotpot.npy'
    if os.path.exists(tfidf_hotpot_filepath):
        tfidf_hotpot = np.load(tfidf_hotpot_filepath).item()
    else:
        print('Hotpot vectorization...')
        tfidf_hotpot = tfidf.transform(qis)
        np.save(tfidf_hotpot_filepath, tfidf_hotpot).item()
    print('Got Hotpot vectors in {:.0f}s'.format(time.time() - start_time))

    # SQuAD: Transform Qs into TFIDF vectors
    start_time = time.time()
    tfidf_squad_filepath = 'data/hotpot-all/tfidf_squad.npy'
    if os.path.exists(tfidf_squad_filepath):
        tfidf_squad = np.load(tfidf_squad_filepath).item()
    else:
        print('SQuAD vectorization...')
        tfidf_squad = tfidf.transform(qks)
        np.save(tfidf_squad_filepath, tfidf_squad)
    print('Got SQuAD vectors in {:.0f}s'.format(time.time() - start_time))

    # HotpotQA-SQuAD TFIDF cosine distances
    start_time = time.time()
    tfidf_cosine_filepath = 'data/hotpot-all/tfidf_cosine.npy'
    if os.path.exists(tfidf_cosine_filepath):
        tfidf_cosine = np.load(tfidf_cosine_filepath)
    else:
        print('Cosine(Hotpot Qs, SQuAD Qs)...')
        num_chunks = 240
        with ThreadPool(48) as p:
            partial_cosine_similarity = partial(cosine_similarity_chunk, X=tfidf_hotpot, Y=tfidf_squad, num_chunks=num_chunks)
            tfidf_cosine = np.hstack(list(tqdm(p.imap(partial_cosine_similarity, list(range(num_chunks))), total=num_chunks)))
        np.save(tfidf_cosine_filepath, tfidf_cosine)
    print('Got Cosine(Hotpot Qs, SQuAD Qs) in {:.0f}s'.format(time.time() - start_time))

    # Load LM for filtering Q's
    if 'lm' in args.filter_name:
        if 'cond-lm' in args.filter_name:
            model, model_tokenizer, model_args = load_model(
                'checkpoint/tn=squad-questions-cond-lm.mn=gpt2-medium.tbs=8.lr=6.25e-05')
        else:  # Unconditional LM
            model, model_tokenizer, model_args = load_model(
                'checkpoint/tn=squad-questions-lm.mn=openai-gpt.tbs=64.lr=6.25e-5')
        model.eval().to('cuda')

        def eval_prob(text):
            indexed_tokens = model_tokenizer.encode(text)
            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')

            with torch.no_grad():
                predictions = model(tokens_tensor)
                if isinstance(predictions, tuple):
                    predictions, past = predictions

            prob_dists = torch.nn.functional.softmax(predictions, dim=-1)
            input_probs = prob_dists[0].gather(1, tokens_tensor[0].unsqueeze(-1))  # NB: Change for batch size > 1
            return input_probs.mean().item()  # TODO: Compute word-level PPL. Also try TransfoXL (whichever has lower Q PPL)

    # Find and modify NNs for each HotpotQA Q
    data_hotpot_new = {'data': []}
    for i, qi in enumerate(qis):
        if (i % args.num_shards) != args.shard_no:
            continue
        # TODO: Filter n-gram in/out substitutions by n-gram TFIDF weight OR substitution should increase TFIDF by \eps
        # qi_word_weights = tfidf_hotpot[i][tfidf_hotpot[i].nonzero()].tolist()[0]
        # qi_bow = tfidf.inverse_transform(tfidf_hotpot[i])[0].tolist()
        # qi_tokens_sorted_by_weight = np.array(sorted(zip(qi_word_weights, qi_bow), reverse=True))[:, 1]

        qi_words = word_tokenizer(qi)
        qi_unwords = word_untokenizer(qi_words)
        qi_prob = eval_prob(qi_unwords) if 'lm' in args.filter_name else 1.0
        print('Original Q #{}: {} ({:.0%})'.format(i, qi_unwords.capitalize(), qi_prob))

        sorted_q_idxs_squad = tfidf_cosine[i].argsort()[::-1]
        data_hotpot_new['data'].append({
            'paragraphs': [{
                'context': cis[i],
                'original_question': qi,
                'nns': [],
                'qas': []
            }],
            'title': ''
        })
        valid_all_nn_mods = 0
        for k_rank in range(args.top_knn):
            k = sorted_q_idxs_squad[k_rank]
            qk = qks[k]
            data_hotpot_new['data'][-1]['paragraphs'][0]['nns'].append(qk)
            qk_words = word_tokenizer(qk)
            qk_unwords = word_untokenizer(qk_words)
            if 'bi-cond-lm' in args.filter_name:
                qk_prob1 = eval_prob(qi_unwords + ' ' + qk_unwords)
                qk_prob2 = eval_prob(qk_unwords + ' ' + qi_unwords)
                qk_prob = (qk_prob1 + qk_prob2) / 2.
            elif 'cond-lm' in args.filter_name:
                qk_prob = eval_prob(qi_unwords + ' ' + qk_unwords)
            elif 'lm' in args.filter_name:
                qk_prob = eval_prob(qk_unwords)
            elif args.filter_name in {'none', 'all'}:
                qk_prob = 1.0
            else:
                raise NotImplementedError('filter_name {}'.format(args.filter_name))
            print('= NN #{}: {} ({:.0%}) (TFIDF: {:.2f})'.format(k_rank + 1, qk_unwords.capitalize(), qk_prob,
                                                                 tfidf_cosine[i, k]))

            if args.filter_name == 'all':
                continue

            max_ngram_size = 3
            valid_nn_mods = 0
            for qi_ngram_size in range(1, max_ngram_size + 1):
                for qk_ngram_size in range(1, max_ngram_size + 1):  # range(max_ngram_size + 1) to include insertions
                    for qi_start_pos in range(len(qi_words) - qi_ngram_size):  # Excludes '?'
                        for qk_start_pos in range(len(qk_words) - qk_ngram_size):  # Excludes '?'
                            qkp_words = qk_words[:qk_start_pos] + \
                                         qi_words[qi_start_pos: qi_start_pos + qi_ngram_size] + \
                                         qk_words[qk_start_pos + qk_ngram_size:]
                            qkp_unwords = word_untokenizer(qkp_words)

                            if 'bi-cond-lm' in args.filter_name:
                                qkp_prob1 = eval_prob(qi_unwords + ' ' + qkp_unwords)
                                qkp_prob2 = eval_prob(qkp_unwords + ' ' + qi_unwords)
                                qkp_prob = (qkp_prob1 + qkp_prob2) / 2.
                                if (qkp_prob1 < qk_prob1) or (qkp_prob2 < qk_prob2):
                                    continue
                            elif 'cond-lm' in args.filter_name:
                                qkp_prob = eval_prob(qi_unwords + ' ' + qkp_unwords)
                            elif 'lm' in args.filter_name:
                                qkp_prob = eval_prob(qkp_unwords)
                            else:  # No filtering
                                qkp_prob = 1.0

                            if qkp_prob < qk_prob:
                                continue

                            print('== {} ({:.0%})'.format(qkp_unwords.capitalize(), qkp_prob))
                            data_hotpot_new['data'][-1]['paragraphs'][0]['qas'].append({
                                'question': qkp_unwords,
                                'answers': [[] for _ in range(len(cis[i]))],
                                'id': idis[i] + '-' + str(valid_all_nn_mods)
                            })
                            valid_nn_mods += 1
                            valid_all_nn_mods += 1
            print('**** {} Valid NN #{} Modifications Found!'.format(valid_nn_mods, k_rank))
        print('**** {} Total Valid NN Modifications Found!'.format(valid_all_nn_mods))
    with open('data/hotpot-all/test.json', 'w') as f:
        json.dump(data_hotpot_new, f)

    return


if __name__ == '__main__':
    main()
