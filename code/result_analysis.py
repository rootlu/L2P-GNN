import argparse
import pickle
import numpy as np
import os
import tensorflow as tf


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def analysis(args):
    # construct result dict
    parent_result_path = '../res/'+ args.dataset + '/'

    result_dict = {}
    for seed_result_path in ["finetune_seed" + str(i) for i in range(args.times)]:
        result_dict[seed_result_path] = {}
        seed_result_names = os.listdir(os.path.join(parent_result_path, seed_result_path))
        # filtered_seed_result_names = seed_result_names  # [n for n in seed_result_names if split in n]
        filtered_seed_result_names = [f for f in seed_result_names if '.' not in f and '300' in f]  # [n for n in seed_result_names if split in n]
        for name in filtered_seed_result_names:
            with open(os.path.join(parent_result_path, seed_result_path, name), "rb") as f:
                result = pickle.load(f)
                result['train'] = result['train']
                result['val'] = result['val']
                result['test_easy'] = result['test_easy']
                result['test_hard'] = result['test_hard']
            result_dict[seed_result_path][name] = result

    # top_k = 40
    best_result_dict = {}  # dict[SEED#][experiment][test_easy/test_hard] = np.array, dim top_k classes
    for seed in result_dict:
        # print(seed)
        best_result_dict[seed] = {}
        for experiment in result_dict[seed]:
            # print(experiment)
            best_result_dict[seed][experiment] = {}
            # val = result_dict[seed][experiment]["val"][:, :top_k]  # look at the top k classes
            val = result_dict[seed][experiment]["val"]
            val_ave = np.average(val, axis=1)
            best_epoch = np.argmax(val_ave)

            # test_easy = result_dict[seed][experiment]["test_easy"][:, :top_k]
            test_easy = result_dict[seed][experiment]["test_easy"]
            test_easy_best = test_easy[best_epoch]

            # test_hard = result_dict[seed][experiment]["test_hard"][:, :top_k]
            test_hard = result_dict[seed][experiment]["test_hard"]
            test_hard_best = test_hard[best_epoch]

            best_result_dict[seed][experiment]["test_easy"] = test_easy_best
            best_result_dict[seed][experiment]["test_hard"] = test_hard_best

    # average across the top k tasks and then average across all the seeds
    mean_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    std_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    for experiment in filtered_seed_result_names:
        # print(experiment)
        mean_result_dict[experiment] = {}
        std_result_dict[experiment] = {}
        test_easy_list = []
        test_hard_list = []
        for seed in best_result_dict:
            test_easy_list.append(best_result_dict[seed][experiment]['test_easy'])
            test_hard_list.append(best_result_dict[seed][experiment]['test_hard'])
        mean_result_dict[experiment]['test_easy'] = np.array(test_easy_list).mean(axis=1).mean()
        mean_result_dict[experiment]['test_hard'] = np.array(test_hard_list).mean(axis=1).mean()
        std_result_dict[experiment]['test_easy'] = np.array(test_easy_list).mean(axis=1).std()
        std_result_dict[experiment]['test_hard'] = np.array(test_hard_list).mean(axis=1).std()

    # results test hard
    sorted_test_hard = sorted(mean_result_dict.items(), key=lambda kv: kv[1]['test_hard'], reverse=True)
    for k, _ in sorted_test_hard:
        print(k)
        print('{} +- {}'.format(mean_result_dict[k]['test_hard'] * 100, std_result_dict[k]['test_hard'] * 100))
        print("")


def get_test_acc(event_file):
    val_auc_list = np.zeros(100)
    test_auc_list = np.zeros(100)
    for e in list(tf.train.summary_iterator(event_file)):
        if len(e.summary.value) == 0:
            continue
        if e.summary.value[0].tag == "data/val_auc":
            val_auc_list[e.step - 1] = e.summary.value[0].simple_value
        if e.summary.value[0].tag == "data/test_auc":
            test_auc_list[e.step - 1] = e.summary.value[0].simple_value

    best_epoch = np.argmax(val_auc_list)

    return test_auc_list[best_epoch]


def analysis_chem(args):
    parent_result_path = '../res/' + args.dataset + '/'
    test_auc = []
    for i, seed in enumerate(range(args.times)):
        dir_name = parent_result_path + "finetune_seed" + str(seed) + "/" + args.file_name
        # print(dir_name)
        file_in_dir = os.listdir(dir_name)
        event_file_list = []
        for f in file_in_dir:
            if "events" in f:
                event_file_list.append(f)

        event_file = event_file_list[0]
        test_auc.append(get_test_acc(dir_name + "/" + event_file))

    print('\n{} +- {}'.format(np.array(test_auc).mean()*100, np.array(test_auc).std()*100))


def analysis_dblp(args):
    # construct result dict
    parent_result_path = '../res/'+ args.dataset + '/'

    result_dict = {}
    for seed_result_path in ["finetune_seed" + str(i) for i in range(args.times)]:
        result_dict[seed_result_path] = {}
        seed_result_names = os.listdir(os.path.join(parent_result_path, seed_result_path))
        filtered_seed_result_names = [f for f in seed_result_names
                                      if '.' not in f
                                      and '300' in f
                                      and 'f1' in f]
        for name in filtered_seed_result_names:
            with open(os.path.join(parent_result_path, seed_result_path, name), "rb") as f:
                result = pickle.load(f)
                result['train'] = result['train']
                result['val'] = result['val']
                result['test'] = result['test']
            result_dict[seed_result_path][name] = result

    # top_k = 40
    best_result_dict = {}  # dict[SEED#][experiment][test_easy/test_hard] = np.array, dim top_k classes
    for seed in result_dict:
        # print(seed)
        best_result_dict[seed] = {}
        for experiment in result_dict[seed]:
            best_result_dict[seed][experiment] = {}
            val = result_dict[seed][experiment]["val"]
            best_epoch = np.argmax(val)

            test_easy = result_dict[seed][experiment]["test"]
            test_easy_best = test_easy[best_epoch]

            best_result_dict[seed][experiment]["test"] = test_easy_best

    # average across the top k tasks and then average across all the seeds
    mean_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    std_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    for experiment in filtered_seed_result_names:
        # print(experiment)
        mean_result_dict[experiment] = {}
        std_result_dict[experiment] = {}
        test_easy_list = []
        for seed in best_result_dict:
            test_easy_list.append(best_result_dict[seed][experiment]['test'])
        mean_result_dict[experiment]['test'] = np.array(test_easy_list).mean()
        std_result_dict[experiment]['test'] = np.array(test_easy_list).std()

    # results test hard
    sorted_test_hard = sorted(mean_result_dict.items(), key=lambda kv: kv[1]['test'], reverse=True)
    for k, _ in sorted_test_hard:
        print(k)
        print('{} +- {}'.format(mean_result_dict[k]['test'] * 100, std_result_dict[k]['test'] * 100))
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--dataset', type=str, default='dblp',
                        help='dataset')
    parser.add_argument('--file_name', type=str, default='co_adaptation_5_300_gin_10',
                        help='file name')
    parser.add_argument('--times', type=int, default=2,
                        help='running times')
    args = parser.parse_args()
    if 'chem' in args.dataset:
        analysis_chem(args)
    elif args.dataset == 'bio':
        analysis(args)
    elif args.dataset == 'dblp':
        analysis_dblp(args)
