# Classification end-to-end benchmark

## Overview
This is a benchmark that present end-to-end classification use-case. Composing parts of this
benchmark are:
1. GPU-powered data preprocessing using DALI,
2. Image classification using EfficientNet,
3. Inference deployment for Triton Inference Server.

## How to run the benchmark?
Running the benchmark requires two steps:
1. Setup
2. Run
3. (optional) Tuning

In the `Setup` step the script downloads the EfficientNet model, converts it to TensorRT format
and puts together model repository and all the configuration files necessary for
Triton Inference Server. It needs to be run only once for a given rig.

In the `Run` step the script runs a Triton's `perf_analyzer` to benchmark the use-case.
There are handful of parameters that can be tuned for the benchmark, you'd most probably
like to turn these nobs in the `Run` step. Details about tuning the solution are presented
in the relevant section below.

Lastly, the `Tuning` step involves polishing the use-case parameters to find the optimal setup
for a given hardware configuration.

### Setup + run
Running both `Setup` and `Run` is required the first time the benchmark is conducted on a given rig.
Please you the following command and provide the maximum batch size you expect :
```bash
$ cd //dali_backend/benchmarks/BM_efficientnet
$ bash benchmark.sh do_setup MAX_BATCH_SIZE
```

### Run
Running the benchmark without the `Setup` step is possible with the command below. Please note
that the script will assume that the benchmark is properly initialized. If not, unpredictable
things may happen. You've been warned!
```bash
$ cd //dali_backend/benchmarks/BM_efficientnet
$ bash benchmark.sh
```

### Factory reset
Factory resetting benchmark might come in handy if an error occurred during the `Setup` step.
To reset the benchmark, please run:
```bash
$ cd //dali_backend/benchmarks/BM_efficientnet
$ bash reset_benchmark.sh 
```

## Tuning the inference
Since this benchmark is targeted into DALI's usage in inference, we'll be focusing on tuning DALI
rather than the EfficientNet model. For the information about the latter, please consult [Triton's
documentation]().

To obtain optimal performance numbers for your given hardware configuration, the scenario has
to be tuned. DALI provides handful of parameters to manipulate. Here we are presenting details
about where these parameters can be found. More information about the parameters themselves
may be found in the [DALI documentation]().

You may tune the benchmark in the following places:
1. `run-benchmark.sh` script. Using `BATCH_SIZES` and `CONCURRENCY_RANGE` variables you can set the values that will be used for the benchmarking.
2. `dali.py` file. This is the file with DALI model definition. As mentioned earlier, DALI provides
handful on parameters (both on the level of pipeline definition and particular operators).
3. Setup command (In the [Setup + run]() section above). There you can set the max batch size used
for inference. The general rule of setting this value for the purposes of the benchmark
is that is should be as big as possible for your hardware setup.