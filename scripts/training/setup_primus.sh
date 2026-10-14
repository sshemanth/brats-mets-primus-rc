#!/bin/bash
set -e
python3 -m venv ~/primus-env
source ~/primus-env/bin/activate
pip install --upgrade pip
pip install torch
git clone https://github.com/MIC-DKFZ/nnUNet.git ~/nnUNet-primus
pip install -e ~/nnUNet-primus
pip install git+https://github.com/MIC-DKFZ/dynamic-network-architectures.git
python -c "import torch; print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available())"
python -c "from dynamic_network_architectures.architectures.primus import PrimusV3S; print('PrimusV3S ok')"
echo SETUP DONE
