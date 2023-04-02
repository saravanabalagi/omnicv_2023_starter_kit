import argparse
import yaml
from munch import DefaultMunch

from train import train
from infer import infer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args_main = parser.parse_args()

    with open(args_main.config) as f:
        args = DefaultMunch(None, yaml.safe_load(f))
        if args.mode == 'train':
            train(args)
        elif args.mode == 'infer':
            infer(args)
        
    

