echo "===== MiLo GeMM throughput benchmark ====="
echo "Reproduce the results in Fig.9"
echo "Expected output format:
ideal
sms = 0108: ,batch=0001: s=0.00091, TFLOP/s=13.373, GB/s=1462.993, speedup=4.54, memory(MB)=1.34, parametersnum : 3.06
...
"
echo "Test running..."
python utils/MiLo_GeMM_throughput_benchmark.py 