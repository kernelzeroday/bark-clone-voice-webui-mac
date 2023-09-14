# Mac port

Install:
```
conda create -n bark-gui python=3.10
conda activate bark-gui
conda install pytorch::pytorch torchvision torchaudio -c pytorch
pip install poetry
poetry install
python webui.py -enablemps
```
