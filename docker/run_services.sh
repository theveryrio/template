#!/bin/bash
# Author: Chanho Kim <theveryrio@gmail.com>

# Navigate to the logs directory
cd $HOME/template/logs

# Create the mlflow directory if mlflow doesn't exist
if [ ! -d "mlflow" ]; then
    mkdir -p mlflow/mlruns
fi

# Navigate to the mlflow directory
cd mlflow

# Start MLflow UI in the background
mlflow ui --host 0.0.0.0 &

# Start Jupyter Lab
cd $HOME/template
jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''
