# Note: This script is ran from the repo root directory
# Also assumes you have python already installed

python -m venv env
source env/Scripts/activate
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install pytorch-lightning
pip install tensorboard