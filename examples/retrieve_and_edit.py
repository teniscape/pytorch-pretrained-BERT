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
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances_argmin_min
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AffinityPropagation, Birch, MeanShift, estimate_bandwidth

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
    parser.add_argument("--filter_name", default="bi-cond-lm", type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--top_knn", default=10, type=int, required=False, help="# of top nearest neighbors to modify.")
    parser.add_argument("--max_mods_per_nn", default=4, type=int, required=False, help="Max # modified Qs to accept per NN Q.")
    parser.add_argument("--seed", default=42, type=int, required=False, help="Random seed")
    parser.add_argument("--num_shards", default=1, type=int, required=False,
                        help="# of total data splits for distributed eval")
    parser.add_argument("--shard_no", default=0, type=int, required=False, help="Distributed eval data split index.")
    args = parser.parse_args()
    args.filter_name = args.filter_name.lower()
    save_filename = 'data/hotpot-all/split=train-dev.filter_name={}.top_knn={}.max_mods_per_nn={}.num_shards={}.shard_no={}.json'.format(
        args.filter_name, args.top_knn, args.max_mods_per_nn, args.num_shards, args.shard_no)
    print('Saving to', save_filename)

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
            return input_probs.mean().item()

    # Find and modify NNs for each HotpotQA Q
    data_hotpot_new = {'data': []}
    for i, qi in tqdm(enumerate(qis)):
        if (i % args.num_shards) != args.shard_no:
            continue

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
            print('= NN #{}: {} ({:.4%}) (TFIDF: {:.2f})'.format(k_rank + 1, qk_unwords.capitalize(), qk_prob,
                tfidf_cosine[i, k]))

            if args.filter_name == 'all':
                continue

            start_time = time.time()
            # qkp_infos = set([])  # Avoid duplicates
            max_ngram_size = 3
            all_qkp_unwords = set([])
            for qi_ngram_size in range(1, max_ngram_size + 1):
                for qk_ngram_size in range(1, max_ngram_size + 1):  # range(max_ngram_size + 1) to include insertions
                    for qi_start_pos in range(len(qi_words) - qi_ngram_size):  # Excludes '?'
                        for qk_start_pos in range(len(qk_words) - qk_ngram_size):  # Excludes '?'
                            qkp_words = qk_words[:qk_start_pos] + \
                                         qi_words[qi_start_pos: qi_start_pos + qi_ngram_size] + \
                                         qk_words[qk_start_pos + qk_ngram_size:]
                            qkp_unwords = word_untokenizer(qkp_words)
                            all_qkp_unwords.add(qkp_unwords)
            print('= NN #{}: Modifying NNs: {:.2f}s)'.format(k_rank + 1, time.time() - start_time))
            start_time = time.time()

            all_qkp_unwords = list(all_qkp_unwords)
            all_qkp_tfidf = tfidf.transform(all_qkp_unwords)
            all_qi2qkp_tfidf_cosine = cosine_similarity(tfidf_hotpot[i], all_qkp_tfidf)[0]
            all_qk2qkp_tfidf_cosine = cosine_similarity(tfidf_squad[k], all_qkp_tfidf)[0]
            qkp_infos = list(zip(all_qkp_unwords, all_qi2qkp_tfidf_cosine, all_qk2qkp_tfidf_cosine))
            print('= NN #{}: Calculating TFIDFs: {:.2f}s)'.format(k_rank + 1, time.time() - start_time))
            start_time = time.time()

            # qkp_aug_infos = []
            max_lm_evals = 32 * args.max_mods_per_nn
            top_qkp_infos = sorted(qkp_infos, key=lambda x: x[1]-x[2], reverse=True)[:max_lm_evals]
            num_lm_approved_qkps = 0
            for qkp_unwords, qi2qkp_tfidf_cosine, qk2qkp_tfidf_cosine in top_qkp_infos:
                if 'bi-cond-lm' in args.filter_name:
                    qkp_prob1 = eval_prob(qi_unwords + ' ' + qkp_unwords)
                    if qkp_prob1 < qk_prob1:
                        continue
                    qkp_prob2 = eval_prob(qkp_unwords + ' ' + qi_unwords)
                    if qkp_prob2 < qk_prob2:
                        continue
                    qkp_prob = (qkp_prob1 + qkp_prob2) / 2.
                elif 'cond-lm' in args.filter_name:
                    qkp_prob = eval_prob(qi_unwords + ' ' + qkp_unwords)
                elif 'lm' in args.filter_name:
                    qkp_prob = eval_prob(qkp_unwords)
                else:  # No filtering
                    qkp_prob = 1.0

                if qkp_prob < qk_prob:
                    continue

                print('=== TFIDF-qi: {:.2f}, TFIDF-qk: {:.2f}, {:.4%}, {}'.format(
                    qi2qkp_tfidf_cosine, qk2qkp_tfidf_cosine, qkp_prob, qkp_unwords.capitalize()))
                data_hotpot_new['data'][-1]['paragraphs'][0]['qas'].append({
                    'question': qkp_unwords,
                    'answers': [[] for _ in range(len(cis[i]))],
                    'id': idis[i] + '-' + str(len(data_hotpot_new['data'][-1]['paragraphs'][0]['qas']))
                })
                num_lm_approved_qkps += 1
                if num_lm_approved_qkps >= args.max_mods_per_nn:
                    break

                # qkp_aug_infos.append((qkp_unwords, qi2qkp_tfidf_cosine, qk2qkp_tfidf_cosine, qkp_prob))
            print('**** {} Valid NN #{} Modifications Found! (LM Evals: {:.2f}s)'.format(
                num_lm_approved_qkps, k_rank + 1, time.time() - start_time))

            # if len(qkp_aug_infos) == 0:
            #     continue
            #
            # cand_qkp_unwords, cand_qi2qkp_tfidf_cosine, cand_qk2qkp_tfidf_cosine, cand_qkp_prob = zip(*qkp_aug_infos)
            # n_clusters = args.max_mods_per_nn ** 2
            # if len(qkp_aug_infos) > n_clusters:  # Cluster valid NN modifications to reduce number of modified examples
            #     tfidf_valid_nn_mods = tfidf.transform(cand_qkp_unwords)
            #     for algo in [None, 'KMeans: TFIDF-weighted', 'GaussianMixture', 'AffinityPropagation', 'Birch']:
            #         start_time = time.time()
            #         if algo == 'KMeans':
            #             cluster = KMeans(n_clusters=n_clusters, random_state=args.seed)
            #             cluster.fit_predict(tfidf_valid_nn_mods, sample_weight=None)
            #             centroid_nn_indices, _ = pairwise_distances_argmin_min(cluster.cluster_centers_, tfidf_valid_nn_mods)
            #         elif algo == 'KMeans: LM-weighted':  # Very similar to KMeans
            #             cluster = KMeans(n_clusters=n_clusters, random_state=args.seed)
            #             cluster.fit_predict(tfidf_valid_nn_mods, sample_weight=cand_qkp_prob)
            #             centroid_nn_indices, _ = pairwise_distances_argmin_min(cluster.cluster_centers_, tfidf_valid_nn_mods)
            #         elif algo == 'KMeans: TFIDF-weighted':  # Very similar to KMeans
            #             cluster = KMeans(n_clusters=n_clusters, random_state=args.seed)
            #             cluster.fit_predict(tfidf_valid_nn_mods, sample_weight=cand_qi2qkp_tfidf_cosine)
            #             centroid_nn_indices, _ = pairwise_distances_argmin_min(cluster.cluster_centers_, tfidf_valid_nn_mods)
            #         elif algo == 'GaussianMixture':
            #             cluster = GaussianMixture(n_components=n_clusters, random_state=args.seed, covariance_type='diag')
            #             cluster.fit_predict(tfidf_valid_nn_mods.toarray())
            #             centroid_nn_indices, _ = pairwise_distances_argmin_min(cluster.means_, tfidf_valid_nn_mods)
            #         elif algo == 'AffinityPropagation':  # Variable cluster #
            #             cluster = AffinityPropagation()
            #             cluster.fit_predict(tfidf_valid_nn_mods)
            #             centroid_nn_indices = cluster.cluster_centers_indices_
            #         elif algo == 'AffinityPropagation: LM-weighted':  # Variable cluster # (many)
            #             cluster = AffinityPropagation(preference=cand_qkp_prob)
            #             cluster.fit_predict(tfidf_valid_nn_mods)
            #             centroid_nn_indices = cluster.cluster_centers_indices_
            #         elif algo == 'AffinityPropagation: TFIDF-weighted':  # Variable cluster # (many)
            #             cluster = AffinityPropagation(preference=cand_qi2qkp_tfidf_cosine)
            #             cluster.fit_predict(tfidf_valid_nn_mods)
            #             centroid_nn_indices = cluster.cluster_centers_indices_
            #         elif algo == 'Birch':  # Variable # of clusters
            #             cluster = Birch(n_clusters=args.max_mods_per_nn, threshold=0.3)
            #             cluster.fit_predict(tfidf_valid_nn_mods)
            #             centroid_nn_indices, _ = pairwise_distances_argmin_min(cluster.subcluster_centers_, tfidf_valid_nn_mods)
            #         elif algo == 'MeanShift':  # Only 1 cluster, Takes long
            #             estimate_bandwidth_time = time.time()
            #             bandwidth = estimate_bandwidth(tfidf_valid_nn_mods.toarray(), n_jobs=-1)
            #             print('==== bandwidth {} ({:.0f}s)'.format(bandwidth, time.time() - estimate_bandwidth_time))
            #             cluster = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            #             cluster.fit_predict(tfidf_valid_nn_mods.toarray())
            #             centroid_nn_indices, _ = pairwise_distances_argmin_min(cluster.cluster_centers_, tfidf_valid_nn_mods)
            #         elif algo is None:
            #             centroid_nn_indices = np.array(range(len(qkp_aug_infos)))
            #         else:
            #             raise NotImplementedError(algo)
            #         print('=== ({:.0f}s) ({} NNs) {}'.format(time.time() - start_time, centroid_nn_indices.size, algo))
            #
            #         centroid_qkp_aug_infos = [qkp_aug_infos[c] for c in centroid_nn_indices]
            #         for qkp_unwords, qi2qkp_tfidf_cosine, qk2qkp_tfidf_cosine, qkp_prob in sorted(
            #                 centroid_qkp_aug_infos, key=lambda x: x[2]-x[3], reverse=True)[:n_clusters]:
            #             print('=== TFIDF-qk: {:.2f}, TFIDF-qi: {:.2f}, {:.4%}, {}'.format(
            #                 qk2qkp_tfidf_cosine, qi2qkp_tfidf_cosine, qkp_prob, qkp_unwords.capitalize()))
            #             data_hotpot_new['data'][-1]['paragraphs'][0]['qas'].append({
            #                 'question': qkp_unwords,
            #                 'answers': [[] for _ in range(len(cis[i]))],
            #                 'id': idis[i] + '-' + str(len(data_hotpot_new['data'][-1]['paragraphs'][0]['qas']))
            #             })
            #         print('TFIDFi {} -> TFIDFi-TFIDFk {} -> LM {} -> Cluster {} -> Final {}'.format(
            #             len(qkp_infos), len(top_qkp_infos), len(qkp_aug_infos), len(centroid_qkp_aug_infos), args.max_mods_per_nn))
            #     else:
            #         centroid_qkp_aug_infos = qkp_aug_infos
            #         for qkp_unwords, qi2qkp_tfidf_cosine, qk2qkp_tfidf_cosine, qkp_prob in sorted(
            #                 centroid_qkp_aug_infos, key=lambda x: x[2]-x[3], reverse=True):
            #             print('=== TFIDF-qk: {:.2f}, TFIDF-qi: {:.2f}, {:.4%}, {}'.format(
            #                 qk2qkp_tfidf_cosine, qi2qkp_tfidf_cosine, qkp_prob, qkp_unwords.capitalize()))
            #             data_hotpot_new['data'][-1]['paragraphs'][0]['qas'].append({
            #                 'question': qkp_unwords,
            #                 'answers': [[] for _ in range(len(cis[i]))],
            #                 'id': idis[i] + '-' + str(len(data_hotpot_new['data'][-1]['paragraphs'][0]['qas']))
            #             })
            #         print('TFIDFi {} -> TFIDFi-TFIDFk {} -> LM {} -> Final {}'.format(
            #             len(qkp_infos), len(top_qkp_infos), len(qkp_aug_infos), args.max_mods_per_nn))

    with open(save_filename, 'w') as f:
        json.dump(data_hotpot_new, f)

    return


if __name__ == '__main__':
    main()
