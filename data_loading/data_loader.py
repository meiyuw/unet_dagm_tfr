# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

""" Dataset class encapsulates the data loading"""
import multiprocessing
import os
from collections import deque

import numpy as np
import tensorflow as tf
from PIL import Image, ImageSequence


class Dataset:
    """Load, separate and prepare the data for training and prediction"""

    def __init__(self, data_dir, batch_size, fold, augment=False, gpu_id=0, num_gpus=1, seed=0, amp=False):
        if not os.path.exists(data_dir):
            raise FileNotFoundError('Cannot find data dir: {}'.format(data_dir))
        print('data_dir:',data_dir)
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._augment = augment
        self.precision = tf.float16 if amp else tf.float32
        self.data_format = "channels_first"

        self._seed = seed

        #images = self._load_multipage_tiff(os.path.join(self._data_dir, 'train-volume.tif'))
        #masks = self._load_multipage_tiff(os.path.join(self._data_dir, 'train-labels.tif'))
        #self._test_images = \
        #    self._load_multipage_tiff(os.path.join(self._data_dir, 'test-volume.tif'))

        #print('image shape =',images[0].shape)
        #train_indices, val_indices = self._get_val_train_indices(len(images), fold)
        #self._train_images = images[train_indices]
        #self._train_masks = masks[train_indices]
        #self._val_images = images[val_indices]
        #self._val_masks = masks[val_indices]
        self.image_shape = [256, 256, 1]
        self._num_gpus = num_gpus
        self._gpu_id = gpu_id    
        
    @property
    def train_size(self):
        return len(self._train_images)

    @property
    def eval_size(self):
        return len(self._val_images)

    @property
    def test_size(self):
        return len(self._test_images)

    @tf.function
    def _preproc_samples(self, inputs, labels, augment=True):
        """Preprocess samples and perform random augmentations"""
        #inputs = self._normalize_inputs(inputs)
        #labels = self._normalize_labels(labels)

        if self._augment and augment:
            # Horizontal flip
            horizontal_flip = (
                tf.random.uniform(shape=(), seed=self.seed) > 0.5
            )
            input_image = tf.cond(
                pred=horizontal_flip,
                true_fn=lambda: tf.image.flip_left_right(input_image),
                false_fn=lambda: input_image,
            )

            mask_image = tf.cond(
                pred=horizontal_flip,
                true_fn=lambda: tf.image.flip_left_right(mask_image),
                false_fn=lambda: mask_image,
            )

            n_rots = tf.random.uniform(
                shape=(), dtype=tf.int32, minval=0, maxval=3, seed=self.seed
            )

            if self.image_shape[0] != self.image_shape[1]:
                n_rots = n_rots * 2

            input_image = tf.image.rot90(input_image, k=n_rots)
            mask_image = tf.image.rot90(mask_image, k=n_rots)

            input_image = tf.image.resize_with_crop_or_pad(
                input_image,
                target_height=self.image_shape[0],
                target_width=self.image_shape[1],
            )

            mask_image = tf.image.resize_with_crop_or_pad(
                mask_image,
                target_height=self.image_shape[0],
                target_width=self.image_shape[1],
            )

        if self.data_format == "channels_first":
            input_image = tf.transpose(a=input_image, perm=[2, 0, 1])

        reshaped_mask_image = tf.reshape(mask_image, [-1])

        # handle mixed precision for float variables
        # int variables remain untouched
        if self.mixed_precision:
            input_image = tf.cast(input_image, dtype=tf.float16)
            reshaped_mask_image = tf.cast(
                reshaped_mask_image, dtype=tf.float16
            )

        #print('train image shape =',inputs.shape)
        #return input_image, reshaped_mask_image
        return tf.cast(input_image, self.precision), reshaped_mask_image

    @tf.function
    def _preproc_eval_samples(self, inputs, labels):
        """Preprocess samples and perform random augmentations"""
        inputs = self._normalize_inputs(inputs)
        labels = self._normalize_labels(labels)

        # Bring back labels to network's output size and remove interpolation artifacts
        labels = tf.image.resize_with_crop_or_pad(labels, target_width=388, target_height=388)
        cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
        labels = tf.where(cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels)))

        return tf.cast(inputs, self.precision), labels

    @tf.function
    def _preproc_test_samples(self, inputs):
        inputs = self._normalize_inputs(inputs)
        return tf.cast(inputs, self.precision)

    def train_fn(self, drop_remainder=False):
        """Input function for training"""
        #dataset = tf.data.Dataset.from_tensor_slices(
        #    (self._train_images, self._train_masks))

        file_pattern = os.path.join(self._data_dir, "train/", "train-*.tfrecords")
        file_list = tf.data.Dataset.list_files(
            file_pattern, seed=self._seed, shuffle=True
        )
        dataset = tf.data.TFRecordDataset(file_list)        

        def _parse_records(record):
            feature_description = {
                "image": tf.io.FixedLenSequenceFeature(
                    self.image_shape, tf.float32, allow_missing=True
                ),
                "mask": tf.io.FixedLenSequenceFeature(
                    self.image_shape, tf.float32, allow_missing=True
                ),
            }

            example = tf.io.parse_single_example(record, feature_description)
            input_image = example["image"]
            mask_image = example["mask"]

            input_image = tf.squeeze(input_image, axis=0)
            mask_image = tf.squeeze(mask_image, axis=0)
            return input_image, mask_image
        
        def _resize_augment_images(input_image, mask_image):
            if self._augment:
                horizontal_flip = (
                    tf.random.uniform(shape=(), seed=self._seed) > 0.5
                )
                input_image = tf.cond(
                    pred=horizontal_flip,
                    true_fn=lambda: tf.image.flip_left_right(input_image),
                    false_fn=lambda: input_image,
                )

                mask_image = tf.cond(
                    pred=horizontal_flip,
                    true_fn=lambda: tf.image.flip_left_right(mask_image),
                    false_fn=lambda: mask_image,
                )

                n_rots = tf.random.uniform(
                    shape=(), dtype=tf.int32, minval=0, maxval=3, seed=self._seed
                )

                if self.image_shape[0] != self.image_shape[1]:
                    n_rots = n_rots * 2

                input_image = tf.image.rot90(input_image, k=n_rots)
                mask_image = tf.image.rot90(mask_image, k=n_rots)

                input_image = tf.image.resize_with_crop_or_pad(
                    input_image,
                    target_height=self.image_shape[0],
                    target_width=self.image_shape[1],
                )

                mask_image = tf.image.resize_with_crop_or_pad(
                    mask_image,
                    target_height=self.image_shape[0],
                    target_width=self.image_shape[1],
                )

            #if self.data_format == "channels_first":
            #    input_image = tf.transpose(a=input_image, perm=[2, 0, 1])

            reshaped_mask_image = tf.reshape(mask_image, [-1])

            # handle mixed precision for float variables
            # int variables remain untouched
            #if self.mixed_precision:
            #    input_image = tf.cast(input_image, dtype=tf.float16)
            #    reshaped_mask_image = tf.cast(
            #        reshaped_mask_image, dtype=tf.float16
            #    )

            return input_image, reshaped_mask_image        
        
        dataset = dataset.map(
            map_func=_parse_records,
            num_parallel_calls=multiprocessing.cpu_count()//self._num_gpus,
        )
        #self._train_images, self._train_masks = dataset        
        dataset = dataset.cache()    
        dataset = dataset.shuffle(self._batch_size * 3)        
        dataset = dataset.shard(self._num_gpus, self._gpu_id)

        dataset = dataset.map(
            map_func=_resize_augment_images,
            num_parallel_calls=multiprocessing.cpu_count()//self._num_gpus,
        )

        dataset = dataset.repeat()
        #dataset = dataset.shuffle(self._batch_size * 3)
        #dataset = dataset.map(self._preproc_samples,
        #                      num_parallel_calls=multiprocessing.cpu_count()//self._num_gpus)
        dataset = dataset.batch(self._batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(self._batch_size)

        return dataset

    def eval_fn(self, count, drop_remainder=False):
        """Input function for training"""
        #dataset = tf.data.Dataset.from_tensor_slices(
        #    (self._train_images, self._train_masks))

        file_pattern = os.path.join(self._data_dir, "test/", "test-*.tfrecords")
        file_list = tf.data.Dataset.list_files(
            file_pattern, seed=self._seed, shuffle=True
        )
        dataset = tf.data.TFRecordDataset(file_list)        

        def _parse_records(record):
            feature_description = {
                "image": tf.io.FixedLenSequenceFeature(
                    self.image_shape, tf.float32, allow_missing=True
                ),
                "mask": tf.io.FixedLenSequenceFeature(
                    self.image_shape, tf.float32, allow_missing=True
                ),
            }

            example = tf.io.parse_single_example(record, feature_description)
            input_image = example["image"]
            mask_image = example["mask"]

            input_image = tf.squeeze(input_image, axis=0)
            mask_image = tf.squeeze(mask_image, axis=0)
            self._val_images = input_image[0]
            return input_image, mask_image
        
        def _resize_augment_images(input_image, mask_image):
            if self._augment:
                horizontal_flip = (
                    tf.random.uniform(shape=(), seed=self._seed) > 0.5
                )
                input_image = tf.cond(
                    pred=horizontal_flip,
                    true_fn=lambda: tf.image.flip_left_right(input_image),
                    false_fn=lambda: input_image,
                )

                mask_image = tf.cond(
                    pred=horizontal_flip,
                    true_fn=lambda: tf.image.flip_left_right(mask_image),
                    false_fn=lambda: mask_image,
                )

                n_rots = tf.random.uniform(
                    shape=(), dtype=tf.int32, minval=0, maxval=3, seed=self._seed
                )

                if self.image_shape[0] != self.image_shape[1]:
                    n_rots = n_rots * 2

                input_image = tf.image.rot90(input_image, k=n_rots)
                mask_image = tf.image.rot90(mask_image, k=n_rots)

                input_image = tf.image.resize_with_crop_or_pad(
                    input_image,
                    target_height=self.image_shape[0],
                    target_width=self.image_shape[1],
                )

                mask_image = tf.image.resize_with_crop_or_pad(
                    mask_image,
                    target_height=self.image_shape[0],
                    target_width=self.image_shape[1],
                )

            #if self.data_format == "channels_first":
            #    input_image = tf.transpose(a=input_image, perm=[2, 0, 1])

            reshaped_mask_image = tf.reshape(mask_image, [-1])


            return input_image, reshaped_mask_image        
        
        dataset = dataset.map(
            map_func=_parse_records,
            num_parallel_calls=multiprocessing.cpu_count()//self._num_gpus,
        )
        #self._train_images, self._train_masks = dataset        
        dataset = dataset.cache()    
        #dataset = dataset.shuffle(self._batch_size * 3)        
        dataset = dataset.shard(self._num_gpus, self._gpu_id)

        dataset = dataset.map(
            map_func=_resize_augment_images,
            num_parallel_calls=multiprocessing.cpu_count()//self._num_gpus,
        )

        #dataset = dataset.repeat()
        #dataset = dataset.shuffle(self._batch_size * 3)
        #dataset = dataset.map(self._preproc_samples,
        #                      num_parallel_calls=multiprocessing.cpu_count()//self._num_gpus)
        dataset = dataset.batch(self._batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(self._batch_size)

        return dataset

    def test_fn(self, count, drop_remainder=False):
        """Input function for testing"""
        dataset = tf.data.Dataset.from_tensor_slices(
            self._test_images)
        dataset = dataset.repeat(count=count)
        dataset = dataset.map(self._preproc_test_samples)
        dataset = dataset.batch(self._batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(self._batch_size)

        return dataset

    def synth_fn(self):
        """Synthetic data function for testing"""
        inputs = tf.random.truncated_normal((572, 572, 1), dtype=tf.float32, mean=127.5, stddev=1, seed=self._seed,
                                            name='synth_inputs')
        masks = tf.random.truncated_normal((388, 388, 2), dtype=tf.float32, mean=0.01, stddev=0.1, seed=self._seed,
                                           name='synth_masks')

        dataset = tf.data.Dataset.from_tensors((inputs, masks))

        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.batch(self._batch_size)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
