CUDA_VISIBLE_DEVICES=1 python bert.py \
--feature_name TextBody  \
--gpus 1 \
--batch_size 8 \
--gradient_accumulation_steps 4 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--truncation_strategy head \
--lr 1e-5 \
--weight_decay 1e-4 \
--use_schedule \
--model_checkpoint roberta-large


CUDA_VISIBLE_DEVICES=1 python bert.py \
--feature_name TextBody  \
--gpus 1 \
--batch_size 16 \
--gradient_accumulation_steps 2 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--truncation_strategy head \
--lr 1e-5 \
--weight_decay 1e-4 \
--use_schedule \
--model_checkpoint roberta-base

