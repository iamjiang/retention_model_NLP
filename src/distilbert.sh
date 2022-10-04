# CUDA_VISIBLE_DEVICES=2 python distilbert.py \
# --feature_name Full_TextBody  \
# --gpus 2 \
# --batch_size 48 \
# --gradient_accumulation_steps 2 \
# --num_epochs 5 \
# --fp16 \
# --loss_weight \
# --truncation_strategy tail \
# --lr 3e-5 \
# --weight_decay 1e-4 \
# --use_schedule \
# --train_negative_positive_ratio 3 \
# --test_negative_positive_ratio 3 \
# --data Full_TextBody_truncation_tail_bert \
# --keep_probab 0.2



CUDA_VISIBLE_DEVICES=2 python distilbert.py \
--feature_name Latest_TextBody  \
--gpus 2 \
--batch_size 48 \
--gradient_accumulation_steps 2 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--truncation_strategy tail \
--lr 3e-5 \
--weight_decay 0 \
--use_schedule \
--data Latest_TextBody_truncation_tail_bert \
--keep_probab 0.2



# CUDA_VISIBLE_DEVICES=2 python distilbert.py \
# --feature_name Client_TextBody  \
# --gpus 2 \
# --batch_size 48 \
# --gradient_accumulation_steps 2 \
# --num_epochs 5 \
# --fp16 \
# --loss_weight \
# --truncation_strategy tail \
# --lr 3e-5 \
# --weight_decay 1e-4 \
# --use_schedule \
# --train_negative_positive_ratio 3 \
# --test_negative_positive_ratio 3 \
# --data Client_TextBody_truncation_tail_bert \
# --keep_probab 0.2


