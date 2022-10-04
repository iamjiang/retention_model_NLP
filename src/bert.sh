CUDA_VISIBLE_DEVICES=3 python bert_base.py \
--feature_name Latest_TextBody  \
--gpus 3 \
--batch_size 24 \
--gradient_accumulation_steps 2 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--truncation_strategy tail \
--lr 3e-5 \
--weight_decay 0 \
--use_schedule \
--data Latest_TextBody_truncation_tail_bert


CUDA_VISIBLE_DEVICES=3 python bert_base.py \
--feature_name Full_TextBody  \
--gpus 3 \
--batch_size 24 \
--gradient_accumulation_steps 2 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--truncation_strategy tail \
--lr 3e-5 \
--weight_decay 0 \
--use_schedule \
--data Full_TextBody_truncation_tail_bert 


CUDA_VISIBLE_DEVICES=3 python bert_base.py \
--feature_name Client_TextBody  \
--gpus 3 \
--batch_size 24 \
--gradient_accumulation_steps 2 \
--num_epochs 5 \
--fp16 \
--loss_weight \
--truncation_strategy tail \
--lr 3e-5 \
--weight_decay 0 \
--use_schedule \
--data Latest_TextBody_truncation_tail_bert

