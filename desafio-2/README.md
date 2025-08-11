#instalar o ambiente virtual
python3 -m venv venv

#iniciar o ambiente virtual
source venv/bin/activate

#Instalações pip
pip install mlflow

#verificar resultado
mlflow ui
