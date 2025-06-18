# 機械学習プロジェクト環境

Kaggle風のJupyter Notebook環境をDockerで構築するプロジェクトです。

## 環境構成

- **ベースイメージ**: `jupyter/datascience-notebook`
- **Python**: 最新版
- **主要ライブラリ**: TensorFlow, PyTorch, scikit-learn, XGBoost, LightGBM, CatBoost
- **実験管理**: MLflow, Weights & Biases
- **ハイパーパラメータ最適化**: Optuna
- **可視化**: matplotlib, seaborn, plotly

## セットアップ

### 前提条件

- Docker
- Docker Compose

### 環境構築

1. リポジトリをクローン
```bash
git clone <repository-url>
cd ml-assignment
```

2. 必要なディレクトリを作成
```bash
mkdir -p notebooks data models scripts
```

3. Dockerコンテナをビルド・起動
```bash
docker-compose up --build
```

## 使い方

### Jupyter Labへのアクセス

コンテナ起動後、ブラウザで以下のURLにアクセス:
```
http://localhost:8888
```

### ディレクトリ構成

```
ml-assignment/
├── notebooks/     # Jupyter Notebookファイル
├── data/         # データセット
├── models/       # 保存済みモデル
├── scripts/      # Pythonスクリプト
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### 主要な機能

1. **機械学習モデリング**
   - scikit-learn, TensorFlow, PyTorchを使用
   - XGBoost, LightGBM, CatBoostによる勾配ブースティング

2. **データ可視化**
   - matplotlib, seaborn, plotlyによる高品質な可視化
   - インタラクティブなグラフ作成

3. **ハイパーパラメータ最適化**
   - Optunaを使用した効率的な最適化

4. **実験管理**
   - MLflowによるモデル・メトリクス管理
   - Weights & Biasesによる実験追跡

5. **モデル解釈**
   - SHAP, LIMEによる予測説明

### コンテナの操作

```bash
# コンテナ起動
docker-compose up

# コンテナ停止
docker-compose down

# バックグラウンドで起動
docker-compose up -d

# ログ確認
docker-compose logs jupyter
```

### データの配置

- データセットは `data/` フォルダに配置
- ホスト側のファイルがコンテナ内に自動同期されます

### モデルの保存

- 訓練済みモデルは `models/` フォルダに保存
- pickleやjoblibを使用した保存が推奨

## トラブルシューティング

### ポート8888が使用中の場合

`docker-compose.yml` のポート設定を変更:
```yaml
ports:
  - "8889:8888"  # ホスト側のポートを変更
```

### パッケージの追加

1. `requirements.txt` に追加
2. コンテナを再ビルド: `docker-compose up --build`

### Kaggle APIの使用

1. Kaggle APIトークンを取得
2. `notebooks/` フォルダに `kaggle.json` を配置
3. Notebook内でKaggleデータセットをダウンロード可能

## 注意事項

- コンテナを停止すると、コンテナ内の一時的な変更は失われます
- 重要なファイルは必ずボリュームマウントされたディレクトリに保存してください