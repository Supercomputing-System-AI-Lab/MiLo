echo "===== MiLo kernel correctness test ====="
echo "Would get assertion error if the result is wrong."
echo "Expected output format:
test_k_stages_divisibility, would show assertion error if the result is wrong.
batch_size:16, hidden: 512, intermediate: 3072, tile_shape:(64,256)
batch_size:16, hidden: 512, intermediate: 3584, tile_shape:(64,256)
..."

echo "Test running...."
python utils/MiLo_kernel_correctness_test.py 