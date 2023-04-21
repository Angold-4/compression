# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model published in:
J. Ballé, V. Laparra, E.P. Simoncelli (2017):
"End-to-end Optimized Image Compression"
Int. Conf. on Learning Representations (ICLR), 2017
https://arxiv.org/abs/1611.01704

With patches from Victor Xing <victor.t.xing@gmail.com>

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.

This script requires TFC v2 (`pip install tensorflow-compression==2.*`).
"""

import argparse
import glob
import sys
from absl import app
from absl.flags import argparse_flags
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_datasets as tfds
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_optimizer import Lamb
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
import torch.nn.init as init


def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=3)


def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)

class Normalize(nn.Module):
    def __init__(self, factor):
        super(Normalize, self).__init__()
        self.factor = factor

    def forward(self, x):
        return x / self.factor

class Denormalize(nn.Module):
    def __init__(self, factor):
        super(Denormalize, self).__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor

class GDN(nn.Module):
    def __init__(self,
                 inverse=False,
                 rectify=False,
                 data_format="channels_last",
                 alpha_parameter=1,
                 beta_parameter=None,
                 gamma_parameter=None,
                 epsilon_parameter=1,
                 alpha_initializer="ones",
                 beta_initializer="ones",
                 gamma_initializer=None,
                 epsilon_initializer="ones"):
        super().__init__()

        self.inverse = inverse
        self.rectify = rectify
        self.alpha_parameter = alpha_parameter
        self.beta_parameter = beta_parameter
        self.gamma_parameter = gamma_parameter
        self.epsilon_parameter = epsilon_parameter

        if gamma_initializer is None:
            self.gamma_initializer = lambda tensor: init.eye_(tensor) * 0.1
        else:
            self.gamma_initializer = gamma_initializer

        self.alpha_initializer = getattr(init, alpha_initializer + "_")
        self.beta_initializer = getattr(init, beta_initializer + "_")
        self.epsilon_initializer = getattr(init, epsilon_initializer + "_")

    def forward(self, inputs):
        if not hasattr(self, 'beta'):
            num_channels = inputs.size(1)
            self.alpha = self.alpha_initializer(torch.empty((), dtype=torch.float32))
            self.beta = self.beta_initializer(torch.empty(num_channels, dtype=torch.float32))
            self.gamma = self.gamma_initializer(torch.empty(num_channels, num_channels, dtype=torch.float32))
            self.epsilon = self.epsilon_initializer(torch.empty((), dtype=torch.float32))

        if self.rectify:
            inputs = F.relu(inputs)

        if self.alpha == 1 and self.rectify:
            norm_pool = inputs
        elif self.alpha == 1:
            norm_pool = torch.abs(inputs)
        elif self.alpha == 2:
            norm_pool = torch.square(inputs)
        else:
            norm_pool = inputs ** self.alpha

        # Ensure the gamma tensor has the same dimension as the input tensor
        gamma_expanded = self.gamma.view(1, *self.gamma.size(), 1, 1)

        # Apply the gamma tensor to the norm pool
        norm_pool = F.conv2d(norm_pool, gamma_expanded, padding=0, groups=num_channels)

        # Add the beta tensor to the norm pool
        norm_pool = norm_pool + self.beta.view(1, -1, 1, 1)

        if self.epsilon == 1:
            pass
        elif self.epsilon == 0.5:
            norm_pool = torch.sqrt(norm_pool)
        else:
            norm_pool = norm_pool ** self.epsilon

        if self.inverse:
            return inputs * norm_pool
        else:
            return inputs / norm_pool

class Encoder(nn.Sequential):
    """The analysis transform."""
    def __init__(self, num_filters):
        super().__init__(
            nn.Conv2d(3, num_filters, kernel_size=9, stride=4, padding=4, bias=True),
            GDN(),
            nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, bias=True),
            GDN(),
            nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, bias=False)
        )

class Decoder(nn.Sequential):
    """The synthesis transform."""
    def __init__(self, num_filters):
        super().__init__(
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            GDN(inverse=True),
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            GDN(inverse=True),
            nn.ConvTranspose2d(num_filters, 3, kernel_size=9, stride=4, padding=4, output_padding=3, bias=True),
            Denormalize(255.0)
        )

class PTAutoencoder(nn.Module):
    def __init__(self, num_filters):
        super(PTAutoencoder, self).__init__()
        self.encoder = Encoder(num_filters)
        self.decoder = Decoder(num_filters)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class TFAutoencoder(tf.keras.Model):
    def __init__(self, num_filters):
        super(TFAutoencoder, self).__init__()
        self.encoder = AnalysisTransform(num_filters)
        self.decoder = SynthesisTransform(num_filters)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

def test_models(input_data, num_filters=64):
    # Convert input data to PyTorch tensor and TensorFlow tensor
    pt_input = torch.tensor(input_data, dtype=torch.float32)
    tf_input = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # Create autoencoders
    pt_autoencoder = PTAutoencoder(num_filters)
    tf_autoencoder = TFAutoencoder(num_filters)

    # Get the outputs
    pt_output = pt_autoencoder(pt_input).detach().numpy()
    tf_output = tf_autoencoder(tf_input).numpy()

    # Compare outputs
    diff = np.abs(pt_output - tf_output)
    print(f'Mean Absolute Difference: {np.mean(diff)}')

class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  def __init__(self, num_filters):
    super().__init__(name="analysis")
    # 1. Input image is divided by 255 to normalize the pixel values between 0 and 1
    self.add(tf.keras.layers.Lambda(lambda x: x / 255.))

    # 2. Convolution layers to extract information from the input image
    self.add(tfc.SignalConv2D(
        num_filters, (9, 9), name="layer_0", corr=True, strides_down=4,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_0")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))


class SynthesisTransform(tf.keras.Sequential):
  """The synthesis transform."""

  def __init__(self, num_filters):
    super().__init__(name="synthesis")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)))
    self.add(tfc.SignalConv2D(
        3, (9, 9), name="layer_2", corr=False, strides_up=4,
        padding="same_zeros", use_bias=True,
        activation=None))
    self.add(tf.keras.layers.Lambda(lambda x: x * 255.))


class BLS2017Model(tf.keras.Model):
  """Main model class."""

  def __init__(self, lmbda, num_filters):
    super().__init__()
    self.lmbda = lmbda

    self.analysis_transform = AnalysisTransform(num_filters)   # encoder
    self.synthesis_transform = SynthesisTransform(num_filters) # decoder

    self.prior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
    self.build((None, None, None, 3))

  def call(self, x, training):
    """Computes rate and distortion losses."""
    entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.prior, coding_rank=3, compression=False)
    x = tf.cast(x, self.compute_dtype)  # TODO(jonycgn): Why is this necessary?

    # VAE

    y = self.analysis_transform(x)          # the encoder 
    y_hat, bits = entropy_model(y, training=training)
    x_hat = self.synthesis_transform(y_hat) # the decoder

    # Total number of bits divided by total number of pixels.
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
    bpp = tf.reduce_sum(bits) / num_pixels
    # Mean squared error across pixels.
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    mse = tf.cast(mse, bpp.dtype)
    # The rate-distortion Lagrangian.
    loss = bpp + self.lmbda * mse
    return loss, bpp, mse

  def train_step(self, x):
    with tf.GradientTape() as tape:
      loss, bpp, mse = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def test_step(self, x):
    loss, bpp, mse = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def predict_step(self, x):
    raise NotImplementedError("Prediction API is not supported.")

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.mse = tf.keras.metrics.Mean(name="mse")

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    # After training, fix range coding tables.
    self.entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.prior, coding_rank=3, compression=True)
    return retval

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
  ])
  def compress(self, x):
    """Compresses an image."""
    # Add batch dimension and cast to float.
    x = tf.expand_dims(x, 0)
    x = tf.cast(x, dtype=self.compute_dtype)
    y = self.analysis_transform(x)
    # Preserve spatial shapes of both image and latents.
    x_shape = tf.shape(x)[1:-1]
    y_shape = tf.shape(y)[1:-1]
    return self.entropy_model.compress(y), x_shape, y_shape

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
  ])
  def decompress(self, string, x_shape, y_shape):
    """Decompresses an image."""
    y_hat = self.entropy_model.decompress(string, y_shape)
    x_hat = self.synthesis_transform(y_hat)
    # Remove batch dimension, and crop away any extraneous padding.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
    # Then cast back to 8-bit integer.
    return tf.saturate_cast(tf.round(x_hat), tf.uint8)


def check_image_size(image, patchsize):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def crop_image(image, patchsize):
  image = tf.image.random_crop(image, (patchsize, patchsize, 3))
  return tf.cast(image, tf.keras.mixed_precision.global_policy().compute_dtype)


def get_dataset(name, split, args):
  """Creates input data pipeline from a TF Datasets dataset."""
  with tf.device("/cpu:0"):
    dataset = tfds.load(name, split=split, shuffle_files=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.filter(
        lambda x: check_image_size(x["image"], args.patchsize))
    dataset = dataset.map(
        lambda x: crop_image(x["image"], args.patchsize))
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def get_custom_dataset(split, args):
  """Creates input data pipeline from custom PNG images."""
  with tf.device("/cpu:0"):
    files = glob.glob(args.train_glob)
    if not files:
      raise RuntimeError(f"No training images found with glob "
                         f"'{args.train_glob}'.")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.map(
        lambda x: crop_image(read_png(x), args.patchsize),
        num_parallel_calls=args.preprocess_threads)
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def train(args):
  """Instantiates and trains the model."""
  if args.precision_policy:
    tf.keras.mixed_precision.set_global_policy(args.precision_policy)
  if args.check_numerics:
    tf.debugging.enable_check_numerics()

  model = BLS2017Model(args.lmbda, args.num_filters)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
  )

  if args.train_glob:
    train_dataset = get_custom_dataset("train", args)
    validation_dataset = get_custom_dataset("validation", args)
  else:
    train_dataset = get_dataset("clic", "train", args)
    validation_dataset = get_dataset("clic", "validation", args)
  validation_dataset = validation_dataset.take(args.max_validation_steps)

  model.fit(
      train_dataset.prefetch(8),
      epochs=args.epochs,
      steps_per_epoch=args.steps_per_epoch,
      validation_data=validation_dataset.cache(),
      validation_freq=1,
      callbacks=[
          tf.keras.callbacks.TerminateOnNaN(),
          tf.keras.callbacks.TensorBoard(
              log_dir=args.train_path,
              histogram_freq=1, update_freq="epoch"),
          tf.keras.callbacks.BackupAndRestore(args.train_path),
      ],
      verbose=int(args.verbose),
  )
  model.save(args.model_path)


def compress(args):
  """Compresses an image."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  x = read_png(args.input_file)
  tensors = model.compress(x)

  # Write a binary file with the shape information and the compressed string.
  packed = tfc.PackedTensors()
  packed.pack(tensors)
  with open(args.output_file, "wb") as f:
    f.write(packed.string)

  # If requested, decompress the image and measure performance.
  if args.verbose:
    x_hat = model.decompress(*tensors)

    # Cast to float in order to compute metrics.
    x = tf.cast(x, tf.float32)
    x_hat = tf.cast(x_hat, tf.float32)
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
    msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

    # The actual bits per pixel including entropy coding overhead.
    num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
    bpp = len(packed.string) * 8 / num_pixels

    print(f"Mean squared error: {mse:0.4f}")
    print(f"PSNR (dB): {psnr:0.2f}")
    print(f"Multiscale SSIM: {msssim:0.4f}")
    print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
    print(f"Bits per pixel: {bpp:0.4f}")


def decompress(args):
  """Decompresses an image."""
  # Load the model and determine the dtypes of tensors required to decompress.
  model = tf.keras.models.load_model(args.model_path)
  dtypes = [t.dtype for t in model.decompress.input_signature]

  # Read the shape information and compressed string from the binary file,
  # and decompress the image using the model.
  with open(args.input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  tensors = packed.unpack(dtypes)
  x_hat = model.decompress(*tensors)

  # Write reconstructed image out as a PNG file.
  write_png(args.output_file, x_hat)


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report progress and metrics when training or compressing.")
  parser.add_argument(
      "--model_path", default="bls2017",
      help="Path where to save/load the trained model.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model. Note that this "
                  "model trains on a continuous stream of patches drawn from "
                  "the training image dataset. An epoch is always defined as "
                  "the same number of batches given by --steps_per_epoch. "
                  "The purpose of validation is mostly to evaluate the "
                  "rate-distortion performance of the model using actual "
                  "quantization rather than the differentiable proxy loss. "
                  "Note that when using custom training images, the validation "
                  "set is simply a random sampling of patches from the "
                  "training set.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--train_glob", type=str, default=None,
      help="Glob pattern identifying custom training data. This pattern must "
           "expand to a list of RGB images in PNG format. If unspecified, the "
           "CLIC dataset from TensorFlow Datasets is used.")
  train_cmd.add_argument(
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  train_cmd.add_argument(
      "--train_path", default="/tmp/train_bls2017",
      help="Path where to log training metrics for TensorBoard and back up "
           "intermediate model checkpoints.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training and validation.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training and validation.")
  train_cmd.add_argument(
      "--epochs", type=int, default=1000,
      help="Train up to this number of epochs. (One epoch is here defined as "
           "the number of steps given by --steps_per_epoch, not iterations "
           "over the full training dataset.)")
  train_cmd.add_argument(
      "--steps_per_epoch", type=int, default=1000,
      help="Perform validation and produce logs after this many batches.")
  train_cmd.add_argument(
      "--max_validation_steps", type=int, default=16,
      help="Maximum number of batches to use for validation. If -1, use one "
           "patch from each image in the training set.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
  train_cmd.add_argument(
      "--precision_policy", type=str, default=None,
      help="Policy for `tf.keras.mixed_precision` training.")
  train_cmd.add_argument(
      "--check_numerics", action="store_true",
      help="Enable TF support for catching NaN and Inf in tensors.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help=f"Output filename (optional). If not provided, appends '{ext}' to "
             f"the input filename.")

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".tfci"
    compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  # app.run(main, flags_parser=parse_args)
    for i in range(100):
        input_data = np.random.rand(1, 3, 256, 256)
        test_models(input_data)
