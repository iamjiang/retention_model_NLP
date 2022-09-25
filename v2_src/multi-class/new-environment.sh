conda create -n test python=3.7
conda activate test
pip install ipykernel
python -m ipykernel install --user --name test
