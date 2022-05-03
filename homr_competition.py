import argparse
import datetime
import os
import re
import tensorflow as tf

from homr_dataset import HOMRDataset
from homr_model import Model
from homr_dataset_utility import create_dataset

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

# Define model parameters defaults

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=25, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--l2", default=0.0001, type=float, help="Weight decay")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate")
parser.add_argument("--rnn_cell_dim", default=128, type=int, help="RNN cell dimension.")
parser.add_argument("--rnn_layers", default=3, type=int, help="Number of RNN layers")
parser.add_argument("--conv_dim", default=128, type=int, help="Conv layers dimension.")
parser.add_argument("--conv_layers", default=3, type=int, help="Number of convolutional layers")
parser.add_argument("--ctc_beam", default=1, type=int, help="Number of sequences for beam search")
parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout rate")
parser.add_argument("--target_img_H", default=110, type=int, help="Adjusts imgs to this height")
parser.add_argument("--target_img_W", default=1720, type=int, help="Adjusts imgs to this width")


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    homr = HOMRDataset()

    train = create_dataset(homr, 'train', args)
    dev = create_dataset(homr, 'dev', args),
    test = create_dataset(homr, 'test', args)

    # Create model
    model = Model(args)
    model.summary()

    # Train and log
    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    # Generate test set annotations
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(test)

        for sequence in predictions:
            print(" ".join(homr.MARKS[mark] for mark in sequence), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
