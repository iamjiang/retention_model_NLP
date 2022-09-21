CUDA_VISIBLE_DEVICES=3 python bert.py \
--feature_name TextBody  \
--gpus 3 \
--batch_size 12 \
--gradient_accumulation_steps 4 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--truncation_strategy head \
--lr 2e-5 \
--weight_decay 1e-4 \
--use_schedule \
--model_checkpoint bert-large-uncased



