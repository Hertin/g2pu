#!/usr/bin/env python3
from ..tasks.g2p import G2PTask


def get_parser():
    parser = G2PTask.get_parser()
    return parser


def main(cmd=None):
    r"""G2P training.

    Example:

        % python g2p_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python g2p_train.py --config conf/train_g2p.yaml
    """
    G2PTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
