

srun -K --ntasks=2 --gpus-per-task=1 --cpus-per-gpu=6 -p V100-32GB --container-mounts=/netscratch:/netscratch,/ds2:/ds2,`pwd`:`pwd` \
--container-image=/netscratch/enroot/dlcc_pytorch_19.12.1.sqsh --container-workdir=`pwd` \
--export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" python yfcc_slurm.py
