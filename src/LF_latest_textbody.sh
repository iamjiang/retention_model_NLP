# CUDA_VISIBLE_DEVICES=1 python 04_model_train_longformer_v1.py \
# --feature_name Latest_TextBody  \
# --gpus 1 \
# --batch_size 6 \
# --gradient_accumulation_steps 8 \
# --num_epochs 5 \
# --fp16 \
# --loss_weight \
# --lr 3e-5 \
# --weight_decay 5e-4 \
# --use_schedule \
# --train_negative_positive_ratio 3 \
# --test_negative_positive_ratio 3 \
# --max_length 2000 \
# --data Latest_TextBody_truncation_tail_longformer


CUDA_VISIBLE_DEVICES=1 python 04_model_train_longformer_v1.py \
--feature_name Latest_TextBody  \
--gpus 1 \
--batch_size 6 \
--gradient_accumulation_steps 8 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--lr 3e-5 \
--weight_decay 5e-4 \
--use_schedule \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3 \
--data email_all

