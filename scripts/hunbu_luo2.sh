# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑核
export SGLANG_SET_CPU_AFFINITY=1
# 设置PYTHONPATH
#cd <code_path>
#export PYTHONPATH=${PWD}/python:$PYTHONPATH
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# export PYTHONPATH=/mnt/share/luochen/sglang-npu-nn-glm-fy/python:$PYTHONPATH
export PYTHONPATH=/mnt/share/luochen/glm/sglang/python:$PYTHONPATH
# cd /home/l00890003/codes/sglang-npu-nn-blue-1
# export PYTHONPATH=${PWD}/python:$PYTHONPATH
# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# 网卡
export HCCL_SOCKET_IFNAME=enp48s3u1u1
export GLOO_SOCKET_IFNAME=enp48s3u1u1
# model path
# MODEL_PATH=/mnt/share/weights/GLM-new-quant-w8a8-quarot

# 老的w4a8权重
# MODEL_PATH=/mnt/share/weights/GLM-New-w4a8_mtp
# 新的w4a8权重
MODEL_PATH=/mnt/share/weights/GLM5-w4a8-mtp
# MODEL_PATH=/mnt/share/l00890003/DeepSeek-V3.2-1201-w8a8




P_IP=('141.61.105.131')
P_MASTER="${P_IP[0]}:4567"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

# mtp环境变量
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

#################################### profiling
# export ENABLE_PROFILING=1

# export SGLANG_NPU_USE_MLAPO=1

#### 定位问题使用，测试性能关闭
# export ASCEND_LAUNCH_BLOCKING=1

export HCCL_OP_EXPANSION_MODE=AIV

export SGLANG_NPU_USE_MULTI_STREAM=1

for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
################################### profiling
        echo "${P_IP[$i]}"
        export HCCL_BUFFSIZE=1000
        #export TASK_QUEUE_ENABLE=2
    #    export ASCEND_USE_FIA=1
#        export ENABLE_ASCEND_MOE_NZ=1
        # python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode prefill \
        # --host ${P_IP[$i]} --port 8000 --disaggregation-bootstrap-port 8995 --trust-remote-code \
        # --nnodes 2 --node-rank $i --tp-size 32 --mem-fraction-static 0.8 \
        # --dist-init-addr ${P_MASTER} \
        # --disable-radix-cache \
        # --attention-backend ascend --device npu  --disaggregation-transfer-backend ascend \
        # --max-running-requests 1 --chunked-prefill-size 66560 --max-prefill-tokens 66560 \
        # --disable-overlap-schedule \
        # --dtype bfloat16 \
        # --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3
        python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 16 --nnodes 1 --node-rank 0 \
        --chunked-prefill-size 16384 --max-prefill-tokens 70000 \
        --trust-remote-code \
        --host 127.0.0.1 \
        --mem-fraction-static 0.7 \
        --port 8000 \
        --served-model-name glm-5 \
        --cuda-graph-bs 16 \
        --quantization modelslim \
        --moe-a2a-backend deepep --deepep-mode auto 
        NODE_RANK=$i
        break
    fi
done    


        # --speculative-algorithm NEXTN \
        # --speculative-num-steps 1 \
        # --speculative-eagle-topk 1 \
        # --speculative-num-draft-tokens 2 \
        # --disable-radix-cache
        # 
        # --quantization modelslim
        #   --moe-a2a-backend deepep --deepep-mode auto     
        # --cuda-graph-bs 2 4 
        # --cuda-graph-max-bs 16 \
        # --disable-cuda-graph 
        # --disable-overlap-schedule 

        # --tool-call-parser glm47 \
        # --reasoning-parser glm45 \
        # --speculative-algorithm EAGLE \
        # --speculative-num-steps 3 \
        # --speculative-eagle-topk 1 \
        # --speculative-num-draft-tokens 3 \
        # --moe-a2a-backend deepep --deepep-mode normal  
