CUDA_VISIBLE_DEVICES=1 python CNN.py \
--gpu 1 \
--loss_weight \
--feature_name Full_TextBody \
--train_batch_size 64 \
--test_batch_size 64 \
--gradient_accumulation_steps 2 \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3 \
--n_epochs 10 \
--learning_rate 2e-5 \
--weight_decay 2e-5 \
--keep_probab 0.4 \
--mode static \
--is_pretrain_embedding \
--kernel_heights 3 4 5 \
--out_channels 100 \
--embedding_length 300 


CUDA_VISIBLE_DEVICES=1 python CNN.py \
--gpu 1 \
--loss_weight \
--feature_name Latest_TextBody \
--train_batch_size 64 \
--test_batch_size 64 \
--gradient_accumulation_steps 2 \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3 \
--n_epochs 10 \
--learning_rate 2e-5 \
--weight_decay 2e-5 \
--keep_probab 0.4 \
--mode static \
--is_pretrain_embedding \
--kernel_heights 3 4 5 \
--out_channels 100 \
--embedding_length 300 

CUDA_VISIBLE_DEVICES=1 python CNN.py \
--gpu 1 \
--loss_weight \
--feature_name Client_TextBody \
--train_batch_size 64 \
--test_batch_size 64 \
--gradient_accumulation_steps 2 \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3 \
--n_epochs 10 \
--learning_rate 2e-5 \
--weight_decay 2e-5 \
--keep_probab 0.4 \
--mode static \
--is_pretrain_embedding \
--kernel_heights 3 4 5 \
--out_channels 100 \
--embedding_length 300 

CUDA_VISIBLE_DEVICES=1 python TF-IDF.py \
--gpu 1 \
--loss_weight \
--feature_name Full_TextBody \
--train_batch_size 256 \
--test_batch_size 256 \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3 \
--n_epochs 10 \
--learning_rate 1e-5 \
--weight_decay 1e-5

CUDA_VISIBLE_DEVICES=1 python TF-IDF.py \
--gpu 1 \
--loss_weight \
--feature_name Latest_TextBody \
--train_batch_size 256 \
--test_batch_size 256 \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3 \
--n_epochs 10 \
--learning_rate 1e-5 \
--weight_decay 1e-5

CUDA_VISIBLE_DEVICES=1 python TF-IDF.py \
--gpu 1 \
--loss_weight \
--feature_name Client_TextBody \
--train_batch_size 256 \
--test_batch_size 256 \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio 3 \
--n_epochs 10 \
--learning_rate 1e-5 \
--weight_decay 1e-5

