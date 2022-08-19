# CUDA_VISIBLE_DEVICES=0 python TF-IDF.py \
# --gpu 0 \
# --loss_weight \
# --feature_name Full_TextBody \
# --train_batch_size 256 \
# --test_batch_size 256 \
# --train_negative_positive_ratio 3 \
# --test_negative_positive_ratio 3 \
# --n_epochs 10 \
# --learning_rate 5e-5 \
# --weight_decay 1e-3 \
# --keep_probab 0.5 \
# --max_features 1000

CUDA_VISIBLE_DEVICES=1 python TF-IDF.py \
--gpu 1 \
--loss_weight \
--feature_name Latest_TextBody \
--train_batch_size 256 \
--test_batch_size 256 \
--n_epochs 10 \
--learning_rate 3e-5 \
--weight_decay 0 \
--keep_probab 0.4 \
--undersampling \
--train_negative_positive_ratio 3 \
--test_negative_positive_ratio  3 \
--max_features 1000



# CUDA_VISIBLE_DEVICES=0 python TF-IDF.py \
# --gpu 0 \
# --loss_weight \
# --feature_name Client_TextBody \
# --train_batch_size 256 \
# --test_batch_size 256 \
# --train_negative_positive_ratio 3 \
# --test_negative_positive_ratio 3 \
# --n_epochs 10 \
# --learning_rate 5e-5 \
# --weight_decay 1e-3 \
# --keep_probab 0.5 \
# --max_features 1000



