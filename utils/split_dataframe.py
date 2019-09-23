import os
import argparse
import numpy as np
import pandas as pd


def main(args):

    df_dir = os.getcwd() + '/data/' + args.which_challenge + '/'

    for phase in ['train', 'valid']:
        os.makedirs(df_dir + 'running_dataframes', exist_ok=True)
        df = pd.read_csv(df_dir + phase + '.csv')
        df = df.sample(frac=1)
        df = np.array_split(df, args.n_splits)

        for isplit in range(len(df)):
            df[isplit].to_csv(df_dir + 'running_dataframes/%s%04d.csv'%(phase, isplit), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--which_challenge', type=str, default='2nd_challenge',
                        help='(2nd_challenge) / (3rd_challenge).')

    parser.add_argument('--n_splits', type=int, default=100,
                        help='number of splits for data frame. (100)')

    args = parser.parse_args()

    main(args)
