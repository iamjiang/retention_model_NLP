CUDA_VISIBLE_DEVICES=0 python 04_model_train_longformer_v1.py \
--feature_name Full_TextBody  \
--gpus 0 \
--batch_size 6 \
--gradient_accumulation_steps 8 \
--num_epochs 10 \
--fp16 \
--loss_weight \
--lr 1e-5 \
--weight_decay 1e-4 \
--use_schedule \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3 \
--max_length 2000