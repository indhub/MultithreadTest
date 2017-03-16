# Test concurrent inference 

## Build:
cmake .

make

## Run
MultithreadTest <device_type> <num_threads> <image> <symbol_file> <params_file>

<device_type> - 1 for CPU, 2 for GPU

<num_threads> - Number of inference to run in parallel

<image> - Image to run inference on

<symbol_file> - The .json symbol file for the model to use

<params_file> - The parameters file for the model to use
