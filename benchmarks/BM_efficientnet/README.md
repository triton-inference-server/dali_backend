# Classification end-to-end benchmark

## Overview
This is a benchmark that presents end-to-end classification use-case. Composing parts of this
benchmark are:
1. GPU-powered data preprocessing using DALI,
2. Image classification using EfficientNet,
3. Inference deployment for Triton Inference Server.

## How to run the benchmark?
Running the benchmark requires three steps:
1. Setup
2. Run
3. (optional) Tuning

In the `Setup` step, the script downloads the EfficientNet model (using [TorchHub](https://pytorch.org/hub/) in [DeepLearningExamples repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master)),
converts it to [TensorRT](https://developer.nvidia.com/tensorrt-getting-started)
format and puts together a model repository and all the configuration files necessary for
Triton Inference Server. It needs to be run only once for a given rig.

In the `Run` step, the script runs a Triton's `perf_analyzer` to benchmark the use-case.
There is a handful of parameters that can be tuned for the benchmark, you'd most probably
like to turn these nobs in the `Tuning` step.

Lastly, the `Tuning` step involves polishing the use-case parameters to find the optimal setup
for a given hardware configuration. Details about tuning the solution are presented
in the relevant section below.

### Setup + run
Running both `Setup` and `Run` is required the first time the benchmark is conducted on a given rig.
Please use the following command and provide the maximal expected batch size (please refer to
the section below for more information on the max batch size):
```bash
$ cd <dali_backend-repo-path>/benchmarks/BM_efficientnet
$ bash benchmark.sh do_setup MAX_BATCH_SIZE
```
For example:
```bash
$ cd /home/myuser/Triton/dali_backend/benchmarks/BM_efficientnet
$ bash benchmark.sh do_setup 32
```

### Run
If you performed the `Setup` step before, it is possible to save some time and run only
the `Run` step. You can do it with the command below. Please note, that if the benchmark has
not been initialized properly, unpredictable things may happen. You've been warned.
```bash
$ cd <dali_backend-repo-path>/benchmarks/BM_efficientnet
$ bash benchmark.sh
```

### Factory reset
Factory resetting benchmark might come in handy if an error occurred during the `Setup` step.
To reset the benchmark, please run:
```bash
$ cd <dali_backend-repo-path>/benchmarks/BM_efficientnet
$ bash reset_benchmark.sh 
```

## Tuning the inference
To obtain optimal performance numbers for your given hardware configuration, the scenario has
to be tuned. DALI provides handful of parameters to manipulate. Here we are presenting details
about where these parameters can be found. More information about the parameters themselves
may be found in the [DALI documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html).

You may tune the benchmark in the following places:
1. `run-benchmark.sh` script. Using `BATCH_SIZES` and `CONCURRENCY_RANGE` variables you can set
the values that will be used for the benchmarking.
2. `dali.py` file. This is the file with DALI model definition. As mentioned earlier, DALI provides
handful on parameters (both on the level of pipeline definition and particular operators). It is
out of scope of this document to explain thoroughly how to tune DALI Pipeline. In the EfficientNet
use case, one of the tuning parameters is the `hw_decoder_load` inside the Pipeline definition.
3. Setup command (In the [Setup + run]() section above). There you can set the max batch size used
for inference. The general rule of setting this value for the purposes of this benchmark
is that the batch size should be as big as possible for your hardware setup.
4. `config.pbtxt` files. [DALI Backend documentation](https://github.com/triton-inference-server/dali_backend/blob/main/docs/config.md)
provides information about the Model Configuration typical for DALI. Particularly, the nob to tune
in the `config.pbtxt` file is the `num_threads` parameter, which denotes number of CPU threads used
by DALI. Also, regular Triton's tuning may be leveraged here, such as configuring model
instances per device (CPU/GPU), or tuning the [Dynamic Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher) parameters.

Lastly, this solution is an ensemble of models. Therefore, tuning the E2E use-case shall be
from the perspective of both - DALI model and the EfficientNet model. For the information, how to
tune the latter, please consult [Model Navigator documentation](https://triton-inference-server.github.io/model_navigator/0.7.1/).

## Benchmark result
The result of the benchmark will be captured as a `csv` file. This file will be saved in the
subdirectory with the date and time of running the benchmark (e.g. `bench-20230824_075412`).
In case the benchmark has been run using multiple batch sizes, the results for every batch
size will be dumped into a separate file, with the pattern: `report-<batch_size>.csv`. Additionally,
all the reports will be combined into one output file per benchmark run in the `combined.csv` file.

Since `perf_analyzer` is used for the benchmark, the meaning of all the values in the report
may be found in the [`perf_analyzer` documentation](https://github.com/triton-inference-server/perf_analyzer/blob/main/README.md).