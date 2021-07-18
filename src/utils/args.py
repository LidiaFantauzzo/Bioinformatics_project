import argparse

MODELS = ["bisenetv2", "deeplabv3", "unet", "none"]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random-seed", type=int, default=42, help="random seed")
    parser.add_argument("--num-classes", type=int, default=2, help="number of segmentation classes")

    # Model
    parser.add_argument("--model", type=str, default="bisenetv2", choices=MODELS,help="model type")
    parser.add_argument("--in-channels", type=int, default=4, help="number of input channels of images")
    parser.add_argument("--output-aux", action="store_true", default=True, help="output-aux for bisenetv1 and bisenetv2")

    # ||| Training and Testing Options |||
    parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="train batch size")
    parser.add_argument("--test-batch-size", type=int, default=50, help="test batch size")
    parser.add_argument("--val-interval", type=int, default=10, help="epoch interval for eval")
    parser.add_argument("--dropout", type=bool, default=False, help="if using dropout or not")

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    # Scheduler
    parser.add_argument("--lr-decay-step", type=int, default=100, help="decay step for stepLR")
    parser.add_argument("--lr-decay-factor", type=float, default=0.7, help="decay factor for stepLR")

    return parser
