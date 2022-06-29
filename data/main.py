# Reference : https://github.com/safreita1/malnet-image
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)

import copy

from tqdm import tqdm
from joblib import Parallel, delayed
from process import copy_images


def run(args_og, group, device):
    args = copy.deepcopy(args_og)

    args['devices'] = [device]
    args['group'] = group

    copy_images(args)


def model_experiments():
    from config import args

    devices = args['devices']
    groups = args['groups']

    Parallel(n_jobs=len(groups))(
        delayed(run)(args, group, devices[idx])
        for idx, group in enumerate(tqdm(groups)))


if __name__ == '__main__':
    model_experiments()
