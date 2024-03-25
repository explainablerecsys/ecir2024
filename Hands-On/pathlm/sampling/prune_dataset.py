import argparse

import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str,  help="Path to file to prune")

    args = parser.parse_args()

    paths = set()
    with open(args.filepath) as f:
        for line in f:
            paths.add(line)

    with open(args.filepath.replace('.txt', '_pruned.txt'), 'w' ) as f:
        for path in paths:
            f.write(path)