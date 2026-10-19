#!/bin/bash
set -uo pipefail
D=/data/qwen38-27b-k100ai-int8-opt/agent_workspaces/native_donor_20261002/gateup/parent_compile_v4
IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
docker run --rm --name q38-native-gateup-v4-graph-20261002 --network none --shm-size 1g --device /dev/kfd --device /dev/dri/renderD128 --group-add video -v /opt/hyhal:/opt/hyhal:ro -v /data:/data:ro -v "$D:/probe" -e HIP_VISIBLE_DEVICES=0 -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/dtk/dcc/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib --entrypoint python3 "$IMAGE" -u "$D/run_graph_three_arm_v4.py" --new-so "$D/q38_smallm_directlds_fused_gateup_v4.so" --out /probe/graph_three_arm_v4.json --logdir /probe > "$D/gpu_graph_v4.log" 2>&1
rc=$?
cat "$D/gpu_graph_v4.log"
echo "GPU_PRIMITIVE_EXIT=$rc"
exit "$rc"
