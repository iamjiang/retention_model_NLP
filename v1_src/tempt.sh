# wget http://nlp.stanford.edu/data/glove.6B.zip
# unzip glove*.zip

CUDA_VISIBLE_DEVICES=1 python CNN.py \
--gpu 1 \
--loss_weight \
--feature_name Full_TextBody \
--train_batch_size 32 \
--test_batch_size 32 \
--gradient_accumulation_steps 4 \
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

# CUDA_VISIBLE_DEVICES=0 python TF-IDF.py \
# --gpu 0 \
# --loss_weight \
# --feature_name Latest_TextBody \
# --train_batch_size 256 \
# --test_batch_size 256 \
# --train_negative_positive_ratio 3 \
# --test_negative_positive_ratio 3 \
# --n_epochs 10 \
# --learning_rate 5e-5 \
# --weight_decay 1e-3 \
# --keep_probab 0.4 \
# --max_features 1000

