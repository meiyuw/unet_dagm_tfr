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
"""Training and evaluation losses"""

import tensorflow as tf


# Class Dice coefficient averaged over batch
def dice_coef(predict, target, axis=1, eps=1e-6):
    intersection = tf.reduce_sum(input_tensor=predict * target, axis=axis)
    union = tf.reduce_sum(input_tensor=predict * predict + target * target, axis=axis)
    dice = (2. * intersection + eps) / (union + eps)
    return tf.reduce_mean(input_tensor=dice, axis=0)  # average over batch


def partial_losses(predict, target):
    n_classes = predict.shape[-1]

    flat_logits = tf.reshape(tf.cast(predict, tf.float32),
                             [tf.shape(input=predict)[0], -1, n_classes])
    flat_labels = tf.reshape(target,
                             [tf.shape(input=predict)[0], -1, n_classes])
    #print('partialloss shape:',flat_logits.shape,flat_labels.shape)

    crossentropy_loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                                            labels=flat_labels),
                                       name='cross_loss_ref')

    dice_loss = tf.reduce_mean(input_tensor=1 - dice_coef(tf.keras.activations.softmax(flat_logits, axis=-1),
                                                          flat_labels), name='dice_loss_ref')
    return crossentropy_loss, dice_loss

def build_total_loss(logits, input_image, labels):
    # Get input image and corresponding gt mask.
    #input_image = features
    reshaped_mask_image = labels

    #is_training = mode == tf.estimator.ModeKeys.TRAIN

    # Flatten the logits
    flatten = tf.keras.layers.Flatten(
        dtype="float32"
    )
    reshaped_logits = flatten(logits)

    # Binary Cross-Entropy loss
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        reshaped_mask_image,
        reshaped_logits,
        #loss_collection=None,
        #reduction=Reduction.SUM_OVER_BATCH_SIZE,
    )
    #print('loss shape:',loss.shape,tf.reduce_sum(loss, axis=0).shape)
    
    #if self.log_image_summaries and is_training:
    #    self._write_image_summaries(
    #        logits, input_image, reshaped_mask_image, is_training=True,
    #    )

    return tf.reduce_sum(loss, axis=0)
