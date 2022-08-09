CUDA_VISIBLE_DEVICES=3 python model_train_bert.py \
--feature_name Full_TextBody  \
--gpus 3 \
--batch_size 24 \
--gradient_accumulation_steps 2 \
--num_epochs 10 \
--fp16 \
--loss_weight \
--trucation_strategy tail \
--lr 1e-5 \
--weight_decay 1e-4 \
--use_schedule \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3


CUDA_VISIBLE_DEVICES=3 python model_train_bert.py \
--feature_name Latest_TextBody  \
--gpus 3 \
--batch_size 24 \
--gradient_accumulation_steps 2 \
--num_epochs 10 \
--fp16 \
--loss_weight \
--trucation_strategy tail \
--lr 1e-5 \
--weight_decay 1e-4 \
--use_schedule \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3


CUDA_VISIBLE_DEVICES=3 python model_train_bert.py \
--feature_name Client_TextBody  \
--gpus 3 \
--batch_size 24 \
--gradient_accumulation_steps 2 \
--num_epochs 10 \
--fp16 \
--loss_weight \
--trucation_strategy tail \
--lr 1e-5 \
--weight_decay 1e-4 \
--use_schedule \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3
