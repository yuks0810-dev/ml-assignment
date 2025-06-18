FROM jupyter/datascience-notebook:latest

USER root

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

USER $NB_UID

# 作業ディレクトリを設定
WORKDIR /home/jovyan/work

# Pythonライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kaggleライブラリとその他の便利なツール
RUN pip install --no-cache-dir \
    kaggle \
    optuna \
    lightgbm \
    xgboost \
    catboost \
    plotly \
    dash \
    streamlit \
    mlflow \
    wandb

# Jupyter拡張機能
RUN pip install --no-cache-dir \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server[all]

# JupyterLabのビルド
RUN jupyter lab build

EXPOSE 8888

CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]