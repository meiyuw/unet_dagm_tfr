# UNet Medical Image Segmentation for TensorFlow 2.x

This repository provides a script and recipe to train UNet model that modify the [Nvidia Unet Medical](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical) example to reflect the [Cerebras modelzoo R1.6.0](https://github.com/Cerebras/modelzoo/tree/R_1.6.0/modelzoo/unet) implementation. The dataset is DAGM 2007, which is a synthetic dataset for defect detection on textured surfaces. The horovod features are removed for testing purposes.
 
 
## Table of Contents
 
- [Model overview](#model-overview)
   * [Model architecture](#model-architecture)
   * [Default configuration](#default-configuration)
   * [Feature support matrix](#feature-support-matrix)
     * [Features](#features)
   * [Mixed precision training](#mixed-precision-training)
     * [Enabling mixed precision](#enabling-mixed-precision)
     * [Enabling TF32](#enabling-tf32)
- [Setup](#setup)
   * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
   * [Scripts and sample code](#scripts-and-sample-code)
   * [Parameters](#parameters)
   * [Command-line options](#command-line-options)
   * [Getting the data](#getting-the-data)
   * [Training process](#training-process)
   * [Inference process](#inference-process)
- [Performance](#performance)   
   * [Benchmarking](#benchmarking)
     * [Training performance benchmark](#training-performance-benchmark)
     * [Inference performance benchmark](#inference-performance-benchmark)
 
 
 
## Model overview
 
The UNet model is a convolutional neural network for 2D image segmentation. This repository contains a UNet implementation as described in the original paper [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), without any alteration.
 
This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results  2.2x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.
 
### Model architecture
 
UNet was first introduced by Olaf Ronneberger, Philip Fischer, and Thomas Brox in the paper: [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). UNet allows for seamless segmentation of 2D images, with high accuracy and performance, and can be adapted to solve many different segmentation problems.
 
The following figure shows the construction of the UNet model and its different components. UNet is composed of a contractive and an expanding path, that aims at building a bottleneck in its centermost part through a combination of convolution and pooling operations. After this bottleneck, the image is reconstructed through a combination of convolutions and upsampling. Skip connections are added with the goal of helping the backward flow of gradients in order to improve the training.
 
![UNet](images/unet.png) 
Figure 1. The architecture of a UNet model. Taken from the <a href="https://arxiv.org/abs/1505.04597">UNet: Convolutional Networks for Biomedical Image Segmentation paper</a>.
 
### Default configuration
 
UNet consists of a contractive (left-side) and expanding (right-side) path. It repeatedly applies unpadded convolutions followed by max pooling for downsampling. Every step in the expanding path consists of an upsampling of the feature maps and concatenation with the correspondingly cropped feature map from the contractive path.
 
 
### Feature support matrix
 
The following features are supported by this model:
 
| **Feature** | **UNet Medical** |
|-------------|---------------------|
| Automatic mixed precision (AMP) | Yes |
| Accelerated Linear Algebra (XLA)| Yes |
 
#### Features
 
**Automatic Mixed Precision (AMP)**
 
This implementation of UNet uses AMP to implement mixed precision training. It allows us to use FP16 training with FP32 master weights by modifying just a few lines of code.
 
 
**XLA support (experimental)**
 
XLA is a domain-specific compiler for linear algebra that can accelerate TensorFlow models with potentially no source code changes. The results are improvements in speed and memory usage: most internal benchmarks run ~1.1-1.5x faster after XLA is enabled.
 
 
### Mixed precision training
 
Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using [mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) previously required two steps:
1.  Porting the model to use the FP16 data type where appropriate.    
2.  Adding loss scaling to preserve small gradient values.

This can now be achieved using Automatic Mixed Precision (AMP) for TensorFlow to enable the full [mixed precision methodology](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow) in your existing TensorFlow model code.  AMP enables mixed precision training on Volta and Turing GPUs automatically. The TensorFlow framework code makes all necessary model changes internally.

In TF-AMP, the computational graph is optimized to use as few casts as necessary and maximize the use of FP16, and the loss scaling is automatically applied inside of supported optimizers. AMP can be configured to work with the existing tf.contrib loss scaling manager by disabling the AMP scaling with a single environment variable to perform only the automatic mixed-precision optimization. It accomplishes this by automatically rewriting all computation graphs with the necessary operations to enable mixed precision training and automatic loss scaling.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.

#### Enabling mixed precision
 
This implementation exploits the TensorFlow Automatic Mixed Precision feature. To enable AMP, you simply need to supply the `--amp` flag to the `main.py` script. For reference, enabling the AMP required us to apply the following changes to the code:
 
1. Set Keras mixed precision policy:
   ```python
   if params['use_amp']:
       tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
   ```
 
2. Use loss scaling wrapper on the optimizer:
   ```python
   optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
   if using_amp:
       optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
   ```
 
3. Use scaled loss to calculate gradients:
   ```python
   scaled_loss = optimizer.get_scaled_loss(loss)
   tape = hvd.DistributedGradientTape(tape)
   scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
   gradients = optimizer.get_unscaled_gradients(scaled_gradients)
   optimizer.apply_gradients(zip(gradients, model.trainable_variables))
   ```
 
#### Enabling TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. 

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.

## Setup
 
The following section lists the requirements that you need to meet in order to start training the UNet Medical model.
 
### Requirements
 
This repository contains Dockerfile which extends the TensorFlow NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- TensorFlow 21.02-tf2-py3 [NGC container](https://ngc.nvidia.com/registry/nvidia-tensorflow) with Tensorflow 2.2 or later
-   GPU-based architecture:
    - [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    - [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/)
    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)
 
For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC container registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
- [Running TensorFlow](https://docs.nvidia.com/deeplearning/dgx/tensorflow-release-notes/running.html#running)
 
For those unable to use the TensorFlow NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide
 
To train your model using mixed precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the UNet model on the [EM segmentation challenge dataset](http://brainiac2.mit.edu/isbi_challenge/home). These steps enable you to build the UNet TensorFlow NGC container, train and evaluate your model, and generate predictions on the test data. Furthermore, you can then choose to:
* compare your evaluation accuracy with our [Training accuracy results](#training-accuracy-results),
* compare your training performance with our [Training performance benchmark](#training-performance-benchmark),
* compare your inference performance with our [Inference performance benchmark](#inference-performance-benchmark).
 
For the specifics concerning training and inference, see the [Advanced](#advanced) section.
 
1. Clone the repository.
 
   Executing this command will create your local repository with all the code to run UNet.
  
   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples
   cd DeepLearningExamples/TensorFlow2/Segmentation/UNet_Medical/
   ```
 
2. Build the UNet TensorFlow NGC container.
 
   This command will use the `Dockerfile` to create a Docker image named `unet_tf2`, downloading all the required components automatically.
  
   ```
   docker build -t unet_tf2 .
   ```
  
   The NGC container contains all the components optimized for usage on NVIDIA hardware.
 
3. Start an interactive session in the NGC container to run preprocessing/training/inference.
 
   The following command will launch the container and mount the `./data` directory as a volume to the `/data` directory inside the container, and `./results` directory to the `/results` directory in the container.
  
   ```bash
   mkdir data
   mkdir results
   docker run --runtime=nvidia -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm --ipc=host -v ${PWD}/data:/data -v ${PWD}/results:/results unet_tf2:latest /bin/bash
   ```
  
   Any datasets and experiment results (logs, checkpoints, etc.) saved to `/data` or `/results` will be accessible
   in the `./data` or `./results` directory on the host, respectively.
 
4. Download and preprocess the data.
  
   The UNet script `main.py` operates on data from the [ISBI Challenge](http://brainiac2.mit.edu/isbi_challenge/home), the dataset originally employed in the [UNet paper](https://arxiv.org/abs/1505.04597). The data is available to download upon registration on the website.
    
   Training and test data are composed of 3 multi-page `TIF` files, each containing 30 2D-images (around 30 Mb total). Once downloaded, the data can be used to run the training and benchmark scripts described below, by pointing `main.py` to its location using the `--data_dir` flag.
  
   **Note:** Masks are only provided for training data.
 
5. Start training.
  
   After the Docker container is launched, the training with the [default hyperparameters](#default-parameters) (for example 1/8 GPUs FP32/TF-AMP) can be started with:
  
   ```bash
   bash examples/unet_TRAIN_SINGLE{_TF-AMP}.sh <number/of/gpus> <path/to/dataset> <path/to/checkpoint>
   ```
  
   For example, to run training with full precision (32-bit) on 1 GPU from the project’s folder, simply use:
  
   ```bash
   bash examples/unet_TRAIN_SINGLE.sh 1 /data /results /model
   ```
  
   This script will launch a training on a single fold (fold 0) and store the model’s checkpoint in the <path/to/checkpoint> directory. 
  
   The script can be run directly by modifying flags if necessary, especially the number of GPUs, which is defined after the `-np` flag. Since the test volume does not have labels, 20% of the training data is used for validation in 5-fold cross-validation manner. The number of fold can be changed using `--fold` with an integer in range 0-4. For example, to run with 4 GPUs using fold 1 use:
  
   ```bash
   horovodrun -np 4 python main.py --data_dir /data --model_dir /results --batch_size 1 --exec_mode train --fold 1 --xla --amp
   ```
  
   Training will result in a checkpoint file being written to `./results` on the host machine.
 
6. Start validation/evaluation.
  
   The trained model can be evaluated by passing the `--exec_mode evaluate` flag. Since evaluation is carried out on a validation dataset, the `--fold` parameter should be filled. For example:
  
   ```bash
   python main.py --data_dir /data --model_dir /results --batch_size 1 --exec_mode evaluate --fold 0 --xla --amp
   ```
  
   Evaluation can also be triggered jointly after training by passing the `--exec_mode train_and_evaluate` flag.
 
7. Start inference/predictions.
 
   The trained model can be used for inference by passing the `--exec_mode predict` flag:
  
   ```bash
   python main.py --data_dir /data --model_dir /results --batch_size 1 --exec_mode predict --xla --amp
   ```
  
   Now that you have your model trained and evaluated, you can choose to compare your training results with our [Training accuracy results](#training-accuracy-results). You can also choose to benchmark the performance of your training [Training performance benchmark](#training-performance-benchmark), or [Inference performance benchmark](#inference-performance-benchmark). Following the steps in these sections will ensure that you achieve the same accuracy and performance results as stated in the [Results](#results) section.
 
## Advanced
 
The following sections provide greater details of the dataset, running training and inference, and the training results.
 
### Scripts and sample code
 
In the root directory, the most important files are:
* `main.py`: Serves as the entry point to the application.
* `run.py`: Implements the logic for training, evaluation, and inference.
* `Dockerfile`: Specifies the container with the basic set of dependencies to run UNet.
* `requirements.txt`: Set of extra requirements for running UNet.
 
The `utils/` folder encapsulates the necessary tools to train and perform inference using UNet. Its main components are:
* `cmd_util.py`: Implements the command-line arguments parsing.
* `data_loader.py`: Implements the data loading and augmentation.
* `losses.py`: Implements the losses used during training and evaluation.
* `parse_results.py`: Implements the intermediate results parsing.
* `setup.py`: Implements helper setup functions.
 
The `model/` folder contains information about the building blocks of UNet and the way they are assembled. Its contents are:
* `layers.py`: Defines the different blocks that are used to assemble UNet.
* `unet.py`: Defines the model architecture using the blocks from the `layers.py` script.
 
Other folders included in the root directory are:
* `examples/`: Provides examples for training and benchmarking UNet.
* `images/`: Contains a model diagram.
 
### Parameters
 
The complete list of the available parameters for the `main.py` script contains:
* `--exec_mode`: Select the execution mode to run the model (default: `train`). Modes available:
  * `train` - trains model from scratch.
  * `evaluate` - loads checkpoint (if available) and performs evaluation on validation subset (requires `--fold` other than `None`).
  * `train_and_evaluate` - trains model from scratch and performs validation at the end (requires `--fold` other than `None`).
  * `predict` - loads checkpoint (if available) and runs inference on the test set. Stores the results in `--model_dir` directory.
  * `train_and_predict` - trains model from scratch and performs inference.
* `--model_dir`: Set the output directory for information related to the model (default: `/results`).
* `--log_dir`: Set the output directory for logs (default: None).
* `--data_dir`: Set the input directory containing the dataset (default: `None`).
* `--batch_size`: Size of each minibatch per GPU (default: `1`).
* `--fold`: Selected fold for cross-validation (default: `None`).
* `--max_steps`: Maximum number of steps (batches) for training (default: `1000`).
* `--seed`: Set random seed for reproducibility (default: `0`).
* `--weight_decay`: Weight decay coefficient (default: `0.0005`).
* `--log_every`: Log performance every n steps (default: `100`).
* `--learning_rate`: Model’s learning rate (default: `0.0001`).
* `--augment`: Enable data augmentation (default: `False`).
* `--benchmark`: Enable performance benchmarking (default: `False`). If the flag is set, the script runs in a benchmark mode - each iteration is timed and the performance result (in images per second) is printed at the end. Works for both `train` and `predict` execution modes.
* `--warmup_steps`: Used during benchmarking - the number of steps to skip (default: `200`). First iterations are usually much slower since the graph is being constructed. Skipping the initial iterations is required for a fair performance assessment.
* `--xla`: Enable accelerated linear algebra optimization (default: `False`).
* `--amp`: Enable automatic mixed precision (default: `False`).
 
### Command-line options
 
To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:
```bash
python main.py --help
```
 
The following example output is printed when running the model:
```python main.py --help
usage: main.py [-h]
              [--exec_mode {train,train_and_predict,predict,evaluate,train_and_evaluate}]
              [--model_dir MODEL_DIR] --data_dir DATA_DIR [--log_dir LOG_DIR]
              [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
              [--fold FOLD]
              [--max_steps MAX_STEPS] [--weight_decay WEIGHT_DECAY]
              [--log_every LOG_EVERY] [--warmup_steps WARMUP_STEPS]
              [--seed SEED] [--augment] [--benchmark]
              [--amp] [--xla]
 
UNet-medical
 
optional arguments:
 -h, --help            show this help message and exit
 --exec_mode {train,train_and_predict,predict,evaluate,train_and_evaluate}
                       Execution mode of running the model
 --model_dir MODEL_DIR
                       Output directory for information related to the model
 --data_dir DATA_DIR   Input directory containing the dataset for training
                       the model
 --log_dir LOG_DIR     Output directory for training logs
 --batch_size BATCH_SIZE
                       Size of each minibatch per GPU
 --learning_rate LEARNING_RATE
                       Learning rate coefficient for AdamOptimizer
 --fold FOLD
                       Chosen fold for cross-validation. Use None to disable
                       cross-validation
 --max_steps MAX_STEPS
                       Maximum number of steps (batches) used for training
 --weight_decay WEIGHT_DECAY
                       Weight decay coefficient
 --log_every LOG_EVERY
                       Log performance every n steps
 --warmup_steps WARMUP_STEPS
                       Number of warmup steps
 --seed SEED           Random seed
 --augment             Perform data augmentation during training
 --benchmark           Collect performance metrics during training
 --amp                 Train using TF-AMP
 --xla                 Train using XLA
```
 
 
### Getting the data
 
Datasets: DAGM 2007/Severstal Steel Defect Detection

[DAGM 2007 competition dataset](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)

This is a synthetic dataset for defect detection on textured surfaces (Binary classification, defective vs. not defective). It was originally created for a competition at the 2007 symposium of the DAGM (Deutsche Arbeitsgemeinschaft für Mustererkennung e.V., the German chapter of the International Association for Pattern Recognition).

Please check the [Cerebras model zoo github repo](https://github.com/Cerebras/modelzoo/tree/R_1.6.0/modelzoo/unet/tf#datasets-dagm-2007severstal-steel-defect-detection) for details about how to preprocess the dataset.
 
### Training process
 
The model trains for a total 6,400 batches (6,400 / number of GPUs), with the default UNet setup:
* Adam optimizer with learning rate of 0.0001.
 
This default parametrization is applied when running scripts from the `./examples` directory and when running `main.py` without explicitly overriding these parameters. By default, the training is in full precision. To enable AMP, pass the `--amp` flag. AMP can be enabled for every mode of execution.
 
The default configuration minimizes a function _L = 1 - DICE + cross entropy_ during training.
 
The training can be run directly without using the predefined scripts. The name of the training script is `main.py`. Because of the multi-GPU support, training should always be run with the Horovod distributed launcher like this:
```bash
horovodrun -np <number/of/gpus> python main.py --data_dir /data [other parameters]
```
 
*Note:* When calling the `main.py` script manually, data augmentation is disabled. In order to enable data augmentation, use the `--augment` flag in your invocation.
 
The main result of the training are checkpoints stored by default in `./results/` on the host machine, and in the `/results` in the container. This location can be controlled
by the `--model_dir` command-line argument, if a different location was mounted while starting the container. In the case when the training is run in `train_and_predict` mode, the inference will take place after the training is finished, and inference results will be stored to the `/results` directory.
 
If the `--exec_mode train_and_evaluate` parameter was used, and if `--fold` parameter is set to an integer value of {0, 1, 2, 3, 4}, the evaluation of the validation set takes place after the training is completed. The results of the evaluation will be printed to the console.

### Inference process
 
Inference can be launched with the same script used for training by passing the `--exec_mode predict` flag:
```bash
python main.py --exec_mode predict --data_dir /data --model_dir <path/to/checkpoint> [other parameters]
```
 
The script will then:
* Load the checkpoint from the directory specified by the `<path/to/checkpoint>` directory (`/results`),
* Run inference on the test dataset,
* Save the resulting binary masks in a `TIF` format.
 
## Performance

The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).
 
### Benchmarking
 
The following section shows how to run benchmarks measuring the model performance in training and inference modes.
 
#### Training performance benchmark
 
To benchmark training, run one of the `TRAIN_BENCHMARK` scripts in `./examples/`:
```bash
bash examples/unet_TRAIN_BENCHMARK{_TF-AMP}.sh <number/of/gpus> <path/to/dataset> <path/to/checkpoints> <batch/size>
```
For example, to benchmark training using mixed-precision on 8 GPUs use:
```bash
bash examples/unet_TRAIN_BENCHMARK_TF-AMP.sh 8 <path/to/dataset> <path/to/checkpoint> <batch/size>
```
 
Each of these scripts will by default run 200 warm-up iterations and benchmark the performance during training in the next 800 iterations.
 
To have more control, you can run the script by directly providing all relevant run parameters. For example:
```bash
horovodrun -np <num of gpus> python main.py --exec_mode train --benchmark --augment --data_dir <path/to/dataset> --model_dir <optional, path/to/checkpoint> --batch_size <batch/size> --warmup_steps <warm-up/steps> --max_steps <max/steps>
```
 
At the end of the script, a line reporting the best train throughput will be printed.
 
#### Inference performance benchmark
 
To benchmark inference, run one of the scripts in `./examples/`:
```bash
bash examples/unet_INFER_BENCHMARK{_TF-AMP}.sh <path/to/dataset> <path/to/checkpoint> <batch/size>
```
 
For example, to benchmark inference using mixed-precision:
```bash
bash examples/unet_INFER_BENCHMARK_TF-AMP.sh <path/to/dataset> <path/to/checkpoint> <batch/size>
```
 
Each of these scripts will by default run 200 warm-up iterations and benchmark the performance during inference in the next 400 iterations.
 
To have more control, you can run the script by directly providing all relevant run parameters. For example:
```bash
python main.py --exec_mode predict --benchmark --data_dir <path/to/dataset> --model_dir <optional, path/to/checkpoint> --batch_size <batch/size> --warmup_steps <warm-up/steps> --max_steps <max/steps>
```
 
At the end of the script, a line reporting the best inference throughput will be printed.

 
