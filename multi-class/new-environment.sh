conda create -n multi-class python=3.7
conda activate multi-class
pip install ipykernel
python -m ipykernel install --user --name multi-class

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirement.txt

