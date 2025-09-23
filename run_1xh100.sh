export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

torchrun --standalone --nproc_per_node=1 train_gpt2.py
