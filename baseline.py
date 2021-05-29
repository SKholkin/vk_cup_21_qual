from itertools import groupby

import pandas as pnd
import tqdm
import numpy as np


def load_train(train_file):
    train = pnd.read_csv(train_file)
    train_inv = train.copy()
    train_inv['tmp'] = train_inv['u']
    train_inv['u'] = train_inv['v']
    train_inv['v'] = train_inv['tmp']
    train_inv.drop(['tmp'], axis=1, inplace=True)
    train = pnd.concat([train, train_inv])
    train.sort_values(['u'], ascending=False, inplace=True)
    return train


def get_recs(train_df, u_part):
    cur_u = -1
    arr = []
    col1 = []
    col2 = []
    col3 = []

    for u, v in tqdm.tqdm(zip(train_df['u'], train_df['v']), total=len(train_df)):
        # для каждой пары друзей u и v
        if u != cur_u:
            # v1 и v2 имеют общего друга cur_u
            for v1 in arr:
                for v2 in arr:
                    if v1 % 10 != u_part: continue
                    if v1 >= v2: continue
                    if v1 % 8 != 1 or v2 % 2 != 1: continue
                    col1.append(v1)
                    col2.append(v2)
                    # Adamic/Adar 
                    col3.append(1 / np.log(len(arr)))
            cur_u = u
            arr = []
        arr.append(v)

        # u и v друзья
        if u % 10 == u_part and u < v and v % 2 == 1 and u % 8 == 1:
            col1.append(u)
            col2.append(v)
            col3.append(np.nan)

    # processing of last cur_u
    for v1 in arr:
        for v2 in arr:
            if v1 % 10 != u_part: continue
            if v1 >= v2: continue
            if v1 % 8 != 1 or v2 % 2 != 1: continue
            col1.append(v1)
            col2.append(v2)
            # Adamic/Adar 
            col3.append(1 / np.log(len(arr)))

    # sort first by col2 second by col1
    # and then create t and v for indeces of col1 and col2 respectively
    ind = np.lexsort((col2, col1))
    t = np.array([col1, col2])[:, ind]
    v = np.array(col3)[ind]

    # count friends heuristic
    # k[0] is v1 k[1] is v2 and sum(v for _, _, v in g)) is score
    cf = [(k[0], k[1], sum(v for _, _, v in g)) for k, g in tqdm.tqdm(groupby(zip(t[0], t[1], v), key=lambda x: (x[0], x[1])), total=len(v))]

    # v / log(number of v neighboors)
    result = []

    # create result
    cur_u = -1
    arr = []
    for u, v, val in tqdm.tqdm(cf):
        if np.isnan(val):continue
        if u != cur_u:
            if cur_u != -1:
                arr.sort(key=lambda x: -x[1])
                result.append(str(cur_u) + ": " + ','.join(map(lambda x: str(x[0]), arr[:10])) + "\n")
            cur_u = u
            arr = []
        arr.append((v, val))

    # process last cur_u
    if cur_u != -1:
        arr.sort(key=lambda x: -x[1])
        result.append(str(cur_u) + ": " + ','.join(map(lambda x: str(x[0]), arr[:10])) + "\n")
    return result


def main(train_file):
    train_df = load_train(train_file)

    result = []

    for u_part in range(10):
        print('start part:', u_part)
        result.extend(get_recs(train_df, u_part))
        print(u_part, 'done')
        break

    with open('subm.txt', 'w') as out:
        out.writelines(result)


if __name__ == '__main__':
    main('train.csv')