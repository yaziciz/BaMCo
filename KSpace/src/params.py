import argparse
from typing import Tuple

def get_default_params():
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    return {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}

def parse_args():
    parser = argparse.ArgumentParser()

    # Only keep arguments that are actually used in the codebase

    # knowledge graph files
    parser.add_argument(
        "--knowledge-graph-train",
        type=str,
        default='/mnt/storage1/ziya/VQA/Datasets/VQARAD/KG_VQARAD_Train_Unprocessed.json',
        help="Path to json file with training data",
    )
    parser.add_argument(
        "--knowledge-graph-test",
        type=str,
        default='/mnt/storage1/ziya/VQA/Datasets/VQARAD/KG_VQARAD_Test_Unprocessed.json',
        help="Path to json file with test data",
    )

    parser.add_argument(
        "--workers", type=int, default=24, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=48, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=True,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--report-to",
        default='tensorboard',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )

    args = parser.parse_args()

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params()
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
    return args