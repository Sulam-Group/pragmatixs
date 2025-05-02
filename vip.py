import argparse

from configs import Constants as C
from configs import get_config
from vip_utils.test import test
from vip_utils.train import train

device = C.device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--workdir", type=str, default=C.workdir)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--max_queries", type=int, default=311)
    train_parser.add_argument("--epochs", type=int, default=500)
    train_parser.add_argument("--batch_size", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-04)
    train_parser.add_argument("--tau_start", type=float, default=1.0)
    train_parser.add_argument("--tau_end", type=float, default=0.2)
    train_parser.add_argument("--sampling", type=str, default="random")
    train_parser.add_argument("--dist", action="store_true", default=False)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--max_queries", type=int, default=311)
    test_parser.add_argument("--tau", type=float, default=0.2)
    test_parser.add_argument("--max_test_queries", type=int, default=50)
    return parser.parse_args()


def main(args):
    config_name = args.config
    workdir = args.workdir
    command = args.command

    config = get_config(config_name)
    if command == "train":
        max_queries = args.max_queries
        epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
        tau_start = args.tau_start
        tau_end = args.tau_end
        sampling = args.sampling
        dist = args.dist
        train(
            config,
            max_queries,
            epochs,
            batch_size,
            lr,
            tau_start,
            tau_end,
            sampling,
            dist=dist,
            workdir=workdir,
        )
    if command == "test":
        max_queries = args.max_queries
        tau = args.tau
        max_test_queries = args.max_test_queries
        test(config, max_queries, tau, max_test_queries, workdir=workdir, device=device)


if __name__ == "__main__":
    args = parse_args()
    main(args)
