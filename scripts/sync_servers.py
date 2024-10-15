import argparse
import os
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('--servers', nargs='+')

    args = parser.parse_args()
    return args

def main(args):

    for server in args.servers:
        print('syncing to ', server)
        command = 'rsync  -avz --copy-links\
             --exclude .idea \
             --exclude __pycache__/ \
             --exclude .DS_Store \
             --exclude .envrc \
             --exclude .git \
             --exclude output/ \
             --exclude wandb/ \
             --exclude saved_weights/fine_tune_single_october/all/ \
             ../procthor-training {}:~/'.format(server)

        os.system(command)

if __name__ == '__main__':
    args = parse_args()
    main(args)