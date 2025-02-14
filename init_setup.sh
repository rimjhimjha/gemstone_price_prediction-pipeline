echo [$(date)]:"START"

echo [$(date)]:"creating env with python 3"

conda create --name env python=3.8 -y

echo [$(date)]:"activating the environment"

source activate env  # If using Windows PowerShell, use `conda activate env`

echo [$(date)]:"Installing the dev requirements"

pip install -r req_dev.txt

echo [$(date)]:"END"
