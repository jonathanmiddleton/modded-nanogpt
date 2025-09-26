export RUN_ID=1
torchrun --standalone --nproc_per_node=8 \
    train_350m.py config/pretrain.yml
