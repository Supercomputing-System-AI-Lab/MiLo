echo "===== MiLo kernel customized GeMM ====="
if [ "$#" -lt 4 ]; then
    echo "Here we actually doing GeMM AxB = C where A.shape=[batchsize, weight_input_dimension], B.shape=[weight_input_dimension, weight_output_dimension]"
    echo "Usage: $0 <batchsize> <weight_output_dimension> <weight_input_dimension> <tile_shape>"
    echo "Here we only support the case when (weight_input_dimension, weight_output_dimension) is a multiple of tileshape"
    echo "Now we only support tile_shape in 64,256 , 128,128."
    echo "Example: bash examples/kernel_custom_GeMM.sh 16 4096 14336 128,128"
    exit 1
fi
echo "Test running ..."
python utils/MiLo_kernel_customized.py --batchsize $1 --weight_output_dimension $2 --weight_input_dimension $3 --tile_shape $4
