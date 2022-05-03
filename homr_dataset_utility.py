from homr_dataset import HOMRDataset
import tensorflow as tf
import argparse


def pad_and_resize(target_img_H, target_img_W):
    def process_example(example: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        image, label = example['image'], example['marks']
        resized_img = tf.image.resize(image, [target_img_H, tf.shape(image)[1]])
        return tf.image.pad_to_bounding_box(resized_img, 0, 0, target_img_H, target_img_W), label
    return process_example


def create_dataset(homr: HOMRDataset, name: str, args: argparse.Namespace) -> tf.data.Dataset:
    dataset = getattr(homr, name)
    target_H, target_W = args.target_img_H, args.target_img_W
    dataset = dataset.map(pad_and_resize(target_H, target_W))
    dataset = dataset.shuffle(args.batch_size * 10, seed=args.seed) if name == "train" else dataset
    dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
