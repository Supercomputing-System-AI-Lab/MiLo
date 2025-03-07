
milo_save_dir="$1"
marlin_save_dir="$2"

echo "===== MiLo kernel Mixtral end-to-end Latency  ====="
echo "=====expercted output=====
Marlin: batchsize: 1
latency 0.123 sec 
batchsize: 16
latency 0.135 sec 
batchsize: 32
latency 0.143 sec 

MiLo: batchsize: 1
latency 0.102 sec 
batchsize: 16
latency 0.112 sec 
batchsize: 32
latency 0.113 sec "
echo "=====MiLo output====="
python utils/MiLo_kernel_end_to_end_latency.py --base_dir ${milo_save_dir}

echo "====downloading Marlin ===="
git clone https://github.com/IST-DASLab/marlin.git
cd marlin
pip install .

echo "=====get 4-bit quant model for Marlin ====="
python utils/Marlin_quant_main.py --base_dir ${marlin_save_dir} --model_id Mixtral --dense_rank 3 --kurtosis_flag s1

echo "===== Marlin end-to-end Latency  ====="
python utils/Marlin_end2end_latency.py --base_dir ${marlin_save_dir}