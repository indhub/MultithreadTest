# Test concurrent inference 

## Build:
cmake .

make

## Run
MultiThreadTest <device_type> <num_threads> \<image> <resize?> <symbol_file> <params_file>

<device_type> - 1 for CPU, 2 for GPU

<num_threads> - Number of inference to run in parallel

\<image> - Image to run inference on

<resize?> - 1 to resize the image inside the thread. 0 otherwise.

<symbol_file> - The .json symbol file for the model to use

<params_file> - The parameters file for the model to use
