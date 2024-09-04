cd /home/chence/Workspaces/P3D/Codes/DatProc
conda activate p3ddp
which python

CUDA_VISIBLE_DEVICES=7 python run.20240812.py --data_source "KHS/Training" --img_root_dir /data/K-Hairstyle/Training/rawset/images/0003.rawset --json_root_dir /data/K-Hairstyle/Training/rawset/labels/0003.rawset --save_root_dir /data/K-Hairstyle/Training/rawset_crop/0003.rawset --total_splits 4 --split_index 2
CUDA_VISIBLE_DEVICES=7 python run.20240812.py --data_source "KHS/Validation" --img_root_dir /data/K-Hairstyle/Validation/rawset/images/0003.rawset --json_root_dir /data/K-Hairstyle/Validation/rawset/labels/0003.rawset --save_root_dir /data/K-Hairstyle/Validation/rawset_crop/0003.rawset --total_splits 4 --split_index 2