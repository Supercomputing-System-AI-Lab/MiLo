#!/bin/bash
# Check if at least 4 arguments are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <batchsize> <hidden_dimension> <mlp_intermediate_dimension> <tile_shape>"
    exit 1
fi

echo "===== MiLo GeMM throughput benchmark ====="
python utils/MiLo_GeMM_throughput_benchmark.py 
echo "===== MiLo kernel correctness test ====="
python utils/MiLo_kernel_correctness_test.py 
echo "===== MiLo kernel customized  ====="
python utils/MiLo_kernel_customized.py --batchsize $1 --hidden_dimension $2 --mlp_intermediate_dimension $3 --tile_shape $4