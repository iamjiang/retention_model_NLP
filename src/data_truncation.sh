
python 03_data_truncation.py \
--truncation_strategy tail \
--max_length 2000 \
--model_checkpoint allenai/longformer-base-4096 \
--model_name longformer \
--feature_name Full_TextBody 

python 03_data_truncation.py \
--truncation_strategy tail \
--max_length 2000 \
--model_checkpoint allenai/longformer-base-4096 \
--model_name longformer \
--feature_name Latest_TextBody 

python 03_data_truncation.py \
--truncation_strategy tail \
--max_length 2000 \
--model_checkpoint allenai/longformer-base-4096 \
--model_name longformer \
--feature_name Client_TextBody 


python 03_data_truncation.py \
--truncation_strategy tail \
--max_length 512 \
--model_checkpoint bert-base-uncased \
--model_name bert \
--feature_name Full_TextBody 

python 03_data_truncation.py \
--truncation_strategy tail \
--max_length 512 \
--model_checkpoint bert-base-uncased \
--model_name bert \
--feature_name Latest_TextBody 

python 03_data_truncation.py \
--truncation_strategy tail \
--max_length 512 \
--model_checkpoint bert-base-uncased \
--model_name bert \
--feature_name Client_TextBody 

