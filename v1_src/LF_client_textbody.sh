CUDA_VISIBLE_DEVICES=1 python longformer.py \
--feature_name Client_TextBody  \
--gpus 1 \
--batch_size 6 \
--gradient_accumulation_steps 8 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--lr 3e-5 \
--weight_decay 1e-4 \
--use_schedule \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3 \
--max_length 4096 \
--data Client_TextBody_truncation_tail_longformer \
--frozen_layers 6

