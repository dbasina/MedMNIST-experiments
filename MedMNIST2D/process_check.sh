nvidia-smi
# or more compact per-GPU list
for i in 0 1 2 3; do
  echo "GPU $i:"
  nvidia-smi -i $i --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
done
