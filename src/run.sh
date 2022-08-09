

CUDA_VISIBLE_DEVICES=0 python 04_model_train_longformer_v1.py --feature_name Full_TextBody  --gpus 0 --batch_size 3 --num_epochs 10 --fp16

CUDA_VISIBLE_DEVICES=1 python 04_model_train_longformer_v1.py --feature_name Latest_TextBody  --gpus 1 --batch_size 3 --num_epochs 10 --fp16

CUDA_VISIBLE_DEVICES=2 python 04_model_train_longformer_v1.py --feature_name Client_TextBody  --gpus 2 --batch_size 3 --num_epochs 10 --fp16

accelerate launch 04_model_train_longformer_v1.py --feature_name Full_TextBody  --gpus 1 2 3 --batch_size 3 --num_epochs 6 --fp16




CUDA_VISIBLE_DEVICES=3 python model_train_bert.py --feature_name Latest_TextBody  --gpus 3 --batch_size 16 --gradient_accumulation_steps 4 --num_epochs 10 --fp16 --trucation_strategy tail 


