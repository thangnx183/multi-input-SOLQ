# GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh --batch_size 3 --coco_path coco_data/bosch_crop
# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --num_feature_levels 1 --coco_path coco_data/carpart_pair/crop --batch_size 8 --epochs 100 --lr_drop 80 --masks --with_box_refine --two_stage --resume exps/video-cp/checkpoint0099.pth --output_dir exps/video-cp-mask
# CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --masks --resume pretrain-cp-bbox.pth --num_feature_levels 3 --coco_path coco_data/carpart_pair/crop --batch_size 8 --epochs 100 --lr_drop 80  --with_box_refine --two_stage  --output_dir exps-mask/demo1 --cls_loss_coef 0 --bbox_loss_coef 0 --giou_loss_coef 0
# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --num_feature_levels 1 --coco_path carpart-side/crop --batch_size 8 --epochs 70 --lr_drop 50 --with_box_refine --two_stage --resume exps/video-side-cp/checkpoint0059.pth --output_dir exps/video-side-cp
# CUDA_VISIBLE_DEVICES=2 python main.py --meta_arch solq --masks --coco_path coco_data/carpart --batch_size 4 --epochs 100 --lr_drop 80  --with_box_refine --two_stage --with_vector  --output_dir exps-mask/demo1

# Phase 1 : single mode
# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py \
#        --meta_arch fast_solq \
#        --num_classes 50 \
#        --with_vector \
#        --with_box_refine \
#        --two_stage \
#        --masks \
#        --batch_size 7 \
#        --coco_path coco_data/carpart-lr/images  \
#        --input_mode single \
#        --pretrain exps-optimal-scheduler-2/cp-side-50-cates-fdct-phase-1/checkpoint.pth\
#        --vector_hidden_dim 256 \
#        --vector_loss_coef 0.01 \
#        --output_dir exps-optimal-scheduler-2/cp-side-50-cates-fdct-phase-1 \
#        # --eval \

# Phase 2 : multi mode
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
#        --meta_arch fast_solq \
#        --num_classes 50 \
#        --with_vector \
#        --with_box_refine \
#        --two_stage \
#        --masks \
#        --batch_size 10 \
#        --coco_path coco_data/carpart-side/crop  \
#        --input_mode multi \
#        --pretrained exps-optimal-scheduler/cp-side-50-cates-fdct-phase-2/checkpoint.pth\
#        --vector_hidden_dim 256 \
#        --vector_loss_coef 0.01 \
#        --lr 6e-5 \
#        --output_dir exps-optimal-scheduler/cp-side-50-cates-fdct-phase-3 \

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env main_cp.py \
#        --meta_arch fast_solq \
#        --num_classes 50 \
#        --with_vector \
#        --with_box_refine \
#        --two_stage \
#        --masks \
#        --batch_size 2 \
#        --coco_path coco_data/carpart-side/crop  \
#        --input_mode multi \
#        --resume exps-optimal-scheduler-2/cp-side-50-cates-fdct-phase-2/checkpoint.pth\
#        --vector_hidden_dim 256 \
#        --vector_loss_coef 0.01 \
#        --lr 6e-5 \
#        --output_dir exps-optimal-scheduler-2/cp-side-50-cates-fdct-phase-2 \

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env main_cp.py \
#        --meta_arch fast_solq \
#        --num_classes 50 \
#        --with_vector \
#        --with_box_refine \
#        --two_stage \
#        --masks \
#        --batch_size 7 \
#        --coco_path coco_data/carpart-side/crop  \
#        --input_mode multi \
#        --resume exps-optimal-scheduler-2/cp-side-50-cates-fdct-phase-3-opti-mask/checkpoint.pth\
#        --vector_hidden_dim 512 \
#        --vector_loss_coef 0.01 \
#        --lr 6e-5 \
#        --output_dir exps-optimal-scheduler-2/cp-side-50-cates-fdct-phase-3-opti-mask \

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
#        --meta_arch fast_solq \
#        --num_classes 50 \
#        --with_vector \
#        --with_box_refine \
#        --two_stage \
#        --masks \
#        --batch_size 8 \
#        --coco_path coco_data/carpart-lr/images  \
#        --input_mode single \
#        --pretrain exps-optimal-scheduler-2/cp-side-50-cates-fdct-phase-3-opti-mask/checkpoint.pth\
#        --vector_hidden_dim 512 \
#        --vector_loss_coef 0.015 \
#        --output_dir exps-optimal-scheduler-3-e2e-opt-mask/phase-1-single-mode \

# Phase 2 : multi mode with ema

CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env main_tracker.py \
       --meta_arch fast_solq \
       --num_classes 50 \
       --with_vector \
       --with_box_refine \
       --two_stage \
       --masks \
       --batch_size 4 \
       --epochs 70 \
       --lr_drop 40 \
       --coco_path coco_data/carpart-side/crop  \
       --input_mode multi \
       --pretrain checkpoint0049.pth\
       --vector_hidden_dim 256 \
       --vector_loss_coef 0.01 \
       --output_dir exps-tracker/phase-4-with-noiser \
