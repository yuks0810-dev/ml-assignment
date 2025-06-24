# Curse of Dimensionality Analysis

次元の呪いと機械学習アルゴリズムの性能分析を行うプロジェクトです。

## 実行方法

### 1. Jupyter Notebook環境での実行

```bash
# Jupyter Lab環境を起動
docker-compose up jupyter

# http://localhost:8888 でアクセス
# notebooks/main.ipynb を開いて実行
```

### 2. Pythonスクリプトでの実行

```bash
# Pythonスクリプトを直接実行
docker-compose run --rm python-script
```

### 3. 個別のサービスを実行

```bash
# Jupyter環境のみ起動
docker-compose up jupyter

# バックグラウンドで起動
docker-compose up -d jupyter

# Pythonスクリプトのみ実行
docker-compose run --rm python-script

# 環境停止
docker-compose down
```

## プロジェクト構成

```
ml-assignment/
├── notebooks/          # Jupyter Notebooks
│   └── main.ipynb     # メインの研究ノートブック
├── scripts/           # Pythonスクリプト
│   └── curse_of_dimensionality_analysis.py  # 実行可能スクリプト
├── data/              # データセット（永続化）
├── models/            # 保存されたモデル（永続化）
├── Dockerfile         # ML環境のDockerイメージ
├── docker-compose.yml # サービス構成
└── requirements.txt   # Python依存関係
```

## 研究内容

### 目的
次元の呪いが機械学習アルゴリズムに与える影響を体系的に分析し、高次元データでのアルゴリズム選択指針を提供します。

### 対象アルゴリズム
- SVM (Support Vector Machine)
- k-NN (k-Nearest Neighbors)
- 混合ガウスモデル (Gaussian Mixture Model)
- 線形回帰 (Linear Regression)

### 実験設定
- **次元数**: 10, 50, 100, 200, 500
- **サンプル数**: 500, 1,000, 5,000
- **データセット**: scikit-learnの合成分類データ

### 分析内容
1. **スパース性分析**: ゼロ成分比率と距離分布の可視化
2. **性能評価**: 次元数とサンプル数による性能変化の測定
3. **耐性分析**: アルゴリズム別の次元の呪いに対する耐性評価
4. **実用ガイドライン**: サンプル/次元比率の推奨事項

## 出力結果

実行後、以下のファイルが生成されます：

- `data/experiment_results.csv` - 実験結果データ
- `data/comprehensive_summary.txt` - 包括的分析レポート
- `models/sparsity_analysis.png` - スパース性分析図
- `models/performance_analysis.png` - 性能分析図
- `models/resistance_analysis.png` - 耐性分析図

## 主要な発見

1. **次元の呪いの確認**: 高次元では距離の変動係数が大幅に減少
2. **アルゴリズム別耐性**: 混合ガウスモデルが最も高い次元耐性を示す
3. **データ効率**: 良好な性能には最低50サンプル/次元の比率が必要
4. **実用的推奨**: 用途別の最適アルゴリズム選択指針を提供

## 推奨事項

- **高次元データ**: 混合ガウスモデル
- **少データ**: SVM
- **安定性重視**: 混合ガウスモデル
- **総合最優秀**: 混合ガウスモデル

## 環境構成

- **ベースイメージ**: `jupyter/datascience-notebook`
- **主要ライブラリ**: TensorFlow, PyTorch, scikit-learn, XGBoost, LightGBM, CatBoost
- **実験管理**: MLflow, Weights & Biases
- **ハイパーパラメータ最適化**: Optuna
- **可視化**: matplotlib, seaborn, plotly

## 環境要件

- Docker & Docker Compose
- 8GB以上のRAM推奨
- ポート8888が利用可能であること

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

## 注意事項

- コンテナを停止すると、コンテナ内の一時的な変更は失われます
- 重要なファイルは必ずボリュームマウントされたディレクトリに保存してください