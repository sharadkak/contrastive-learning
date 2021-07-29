# training

srun -K --ntasks=5 --gpus-per-task=1 --cpus-per-gpu=8 -p V100-32GB --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
--container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh --container-workdir=`pwd` \
--export="NCCL_SOCKET_IFNAME=bond0,NCCL_IB_HCA=mlx5,NCCL_IB_TIMEOUT=22,NCCL_IB_DISABLE=1" python imagenet_slurm.py
