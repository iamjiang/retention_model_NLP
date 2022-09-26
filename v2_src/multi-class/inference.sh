CUDA_VISIBLE_DEVICES=2 python model_inference.py \
--feature_name TextBody  \
--gpus 2 \
--batch_size 128 \
--loss_weight \
--model_path /home/ec2-user/SageMaker/retention_model_NLP/v2_src/multi-class/roberta_large_repo
