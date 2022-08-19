CUDA_VISIBLE_DEVICES=0 python longformer.py \
--feature_name Latest_TextBody  \
--gpus 0 \
--batch_size 6 \
--gradient_accumulation_steps 8 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--lr 3e-5 \
--weight_decay 0 \
--use_schedule \
--max_length 4096 \
--data Latest_TextBody_truncation_tail_longformer \
--frozen_layers 6

