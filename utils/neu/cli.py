
import argparse


def setup_fid_cli(subparsers, callbacks):
    from neu.metrics.gan_eval import fid
    parser = fid.get_parser()
    subparsers.add_parser("fid", parents=[parser], conflict_handler='resolve')
    callbacks["fid"] = fid.main


def setup_inception_score_cli(subparsers, callbacks):
    from neu.metrics.gan_eval import inception_score
    parser = inception_score.get_parser()
    subparsers.add_parser("inception_score", parents=[
                          parser], conflict_handler='resolve')
    callbacks["inception_score"] = inception_score.main


def main():
    parser = argparse.ArgumentParser("neu_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    callbacks = {}

    # fid
    setup_fid_cli(subparsers, callbacks)

    # inception score
    setup_inception_score_cli(subparsers, callbacks)

    args = parser.parse_args()

    if args.command not in callbacks:
        raise ValueError(
            f"The command '{args.command}' is not supported. Must be one of {callbacks.keys()}.")

    callbacks[args.command](args)
