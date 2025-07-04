import glob
import random
import itertools
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import csv
import os
import argparse

def make_partition(
    signers: List[int],
    pair_genuine_genuine: List[Tuple[int, int]],
    pair_genuine_forged: List[Tuple[int, int]],
):
    samples = []
    for signer_id in signers:
        sub_pair_genuine_forged = random.sample(pair_genuine_forged, len(pair_genuine_genuine))
        genuine_genuine = list(itertools.zip_longest(pair_genuine_genuine, [], fillvalue=1)) # y = 1
        genuine_genuine = list(map(lambda sample: (signer_id, *sample[0], sample[1]), genuine_genuine))
        samples.extend(genuine_genuine)
        genuine_forged = list(itertools.zip_longest(sub_pair_genuine_forged, [], fillvalue=0)) # y = 0
        genuine_forged = list(map(lambda sample: (signer_id, *sample[0], sample[1]), genuine_forged))
        samples.extend(genuine_forged)
    return samples

def write_csv(file_path, samples):
    with open(file_path, 'wt') as f:
        writer = csv.writer(f)
        writer.writerows(samples)

def prepare_CEDAR(M: int, K: int, random_state=0, data_dir='data/CEDAR'):
    def get_path(row):
        writer_id, x1, x2, y = row
        if y == 1:
            x1 = os.path.join(data_dir, 'full_org', f'original_{writer_id}_{x1}.png')
            x2 = os.path.join(data_dir, 'full_org', f'original_{writer_id}_{x2}.png')
        else:
            x1 = os.path.join(data_dir, 'full_org', f'original_{writer_id}_{x1}.png')
            x2 = os.path.join(data_dir, 'full_forg', f'forgeries_{writer_id}_{x2}.png')
        return x1, x2, y # drop writer_id

    random.seed(random_state)
    signers = list(range(1, K+1))
    num_genuine_sign = 24
    num_forged_sign = 24

    train_signers, test_signers = train_test_split(signers, test_size=K-M)
    pair_genuine_genuine = list(itertools.combinations(range(1, num_genuine_sign+1), 2))
    pair_genuine_forged = list(itertools.product(range(1, num_genuine_sign+1), range(1, num_forged_sign+1)))

    train_samples = make_partition(train_signers, pair_genuine_genuine, pair_genuine_forged)
    train_samples = list(map(get_path, train_samples))
    write_csv(os.path.join(data_dir, 'train.csv'), train_samples)
    test_samples = make_partition(test_signers, pair_genuine_genuine, pair_genuine_forged)
    test_samples = list(map(get_path, test_samples))
    write_csv(os.path.join(data_dir, 'test.csv'), test_samples)


def prepare_BHSig260(M: int, K: int, random_state=0, data_dir='data/BHSig260/Bengali'):
    def get_path(row):
        writer_id, x1, x2, y = row
        if y == 1:
            x1 = os.path.join(data_dir, f'{writer_id:03d}', f'B-S-{writer_id}-G-{x1:02d}.tif')
            x2 = os.path.join(data_dir, f'{writer_id:03d}', f'B-S-{writer_id}-G-{x2:02d}.tif')
        else:
            x1 = os.path.join(data_dir, f'{writer_id:03d}', f'B-S-{writer_id}-G-{x1:02d}.tif')
            x2 = os.path.join(data_dir, f'{writer_id:03d}', f'B-S-{writer_id}-F-{x2:02d}.tif')
        return x1, x2, y # drop writer_id

    random.seed(random_state)
    signers = list(range(1, K+1))
    num_genuine_sign = 24
    num_forged_sign = 30

    train_signers, test_signers = train_test_split(signers, test_size=K-M)
    pair_genuine_genuine = list(itertools.combinations(range(1, num_genuine_sign+1), 2))
    pair_genuine_forged = list(itertools.product(range(1, num_genuine_sign+1), range(1, num_forged_sign+1)))

    train_samples = make_partition(train_signers, pair_genuine_genuine, pair_genuine_forged)
    train_samples = list(map(get_path, train_samples))
    write_csv(os.path.join(data_dir, 'train.csv'), train_samples)
    test_samples = make_partition(test_signers, pair_genuine_genuine, pair_genuine_forged)
    test_samples = list(map(get_path, test_samples))
    write_csv(os.path.join(data_dir, 'test.csv'), test_samples)


def prepare_sign_data(data_dir='sign_data', random_state=2020):
    def get_writers(path):
        return [os.path.basename(d) for d in glob.glob(os.path.join(path, '*')) if not d.endswith('_forg')]

    def make_samples(writers, subset_path):
        all_genuine_pairs = []
        all_forged_pairs = []
        for writer_id in writers:
            genuine_path = os.path.join(subset_path, writer_id)
            forged_path = os.path.join(subset_path, f"{writer_id}_forg")

            genuine_files = glob.glob(os.path.join(genuine_path, '*.*[gG]'))
            forged_files = glob.glob(os.path.join(forged_path, '*.*[gG]'))

            if not genuine_files or not forged_files:
                continue

            # Create pairs
            genuine_pairs = list(itertools.combinations(genuine_files, 2))
            forged_pairs = list(itertools.product(genuine_files, forged_files))
            
            all_genuine_pairs.extend(genuine_pairs)
            all_forged_pairs.extend(forged_pairs)
        
        # Balance and create samples
        if len(all_forged_pairs) > len(all_genuine_pairs):
            all_forged_pairs = random.sample(all_forged_pairs, len(all_genuine_pairs))
        else:
            all_genuine_pairs = random.sample(all_genuine_pairs, len(all_forged_pairs))

        genuine_samples = [(p[0], p[1], 1) for p in all_genuine_pairs]
        forged_samples = [(p[0], p[1], 0) for p in all_forged_pairs]
        
        samples = genuine_samples + forged_samples
        random.shuffle(samples)
        return samples

    random.seed(random_state)
    
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')

    train_writers = get_writers(train_path)
    test_writers = get_writers(test_path)

    train_samples = make_samples(train_writers, train_path)
    test_samples = make_samples(test_writers, test_path)

    write_csv(os.path.join(data_dir, 'train.csv'), train_samples)
    write_csv(os.path.join(data_dir, 'test.csv'), test_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument('--dataset', type=str, default='sign_data', choices=['cedar', 'sign_data', 'bengali', 'hindi'])
    args = parser.parse_args()

    if args.dataset == 'cedar':
        print('Preparing CEDAR dataset..')
        prepare_CEDAR(M=50, K=55)
    elif args.dataset == 'sign_data':
        print('Preparing sign_data dataset..')
        prepare_sign_data(data_dir='sign_data')
    elif args.dataset == 'bengali':
        print('Preparing Bengali dataset..')
        prepare_BHSig260(M=50, K=100, data_dir='data/BHSig260/Bengali')
    elif args.dataset == 'hindi':
        print('Preparing Hindi dataset..')
        prepare_BHSig260(M=100, K=160, data_dir='data/BHSig260/Hindi')
    
    print('Done')