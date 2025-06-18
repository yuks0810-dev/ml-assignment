# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project that provides a Kaggle-style Jupyter Notebook environment using Docker. The project contains a comprehensive study on the curse of dimensionality and machine learning algorithm performance analysis.

## Environment Setup and Management

### Docker Environment Commands
```bash
# Build and start the Jupyter environment
docker-compose up --build

# Start existing environment
docker-compose up

# Start in background
docker-compose up -d

# Stop environment
docker-compose down

# View logs
docker-compose logs jupyter
```

### Access Jupyter Lab
- Access at: http://localhost:8888
- No token/password required (configured for development)

### Adding New Python Packages
1. Add package to `requirements.txt`
2. Rebuild container: `docker-compose up --build`

## Architecture

### Directory Structure
```
ml-assignment/
├── notebooks/     # Jupyter Notebooks (main research content)
├── data/         # Datasets (volume mounted, persistent)
├── models/       # Saved models (volume mounted, persistent)
├── scripts/      # Standalone Python scripts
├── Dockerfile    # Jupyter environment with ML libraries
├── docker-compose.yml  # Service orchestration
└── requirements.txt    # Python dependencies
```

### Key Components

**Main Research Notebook** (`notebooks/main.ipynb`):
- Comprehensive study on curse of dimensionality
- Compares 6 ML algorithms: k-NN, SVM, Random Forest, Logistic Regression, Neural Network, PCA+k-NN
- Analyzes algorithm performance across different dimensions (2-100) and sample sizes (500-5000)
- Provides quantitative analysis of algorithm resistance to high-dimensional data

**Sample Workflow** (`notebooks/sample_ml_workflow.ipynb`):
- Basic ML workflow demonstration
- Uses Iris and California Housing datasets
- Template for new ML experiments

### Docker Configuration
- Base: `jupyter/datascience-notebook:latest`
- Pre-installed ML stack: TensorFlow, PyTorch, scikit-learn, XGBoost, LightGBM, CatBoost
- Experiment tracking: MLflow, Weights & Biases
- Hyperparameter optimization: Optuna
- Data visualization: matplotlib, seaborn, plotly

## Development Workflow

### Working with Notebooks
- All notebook work should be done in `notebooks/` directory
- Data files go in `data/` directory (auto-mounted to container)
- Model artifacts save to `models/` directory (persistent)
- Use `sample_ml_workflow.ipynb` as template for new experiments

### Data Persistence
- Only files in mounted volumes (`notebooks/`, `data/`, `models/`, `scripts/`) persist between container restarts
- Container-only changes are lost on restart

### Troubleshooting
- If port 8888 is busy, modify `docker-compose.yml` ports section
- For package conflicts, check `requirements.txt` versions
- Container logs: `docker-compose logs jupyter`

## Research Capabilities

The environment supports:
- Traditional ML algorithms (scikit-learn ecosystem)
- Deep learning (TensorFlow, PyTorch)
- Gradient boosting (XGBoost, LightGBM, CatBoost)
- Experiment tracking and visualization
- Advanced hyperparameter optimization
- Model interpretability (SHAP, LIME)
- Kaggle dataset integration

## Key Notebooks

**main.ipynb** - Primary research on curse of dimensionality:
- Systematic experimental framework
- Distance distribution analysis in high dimensions
- Algorithm performance degradation measurement
- Data efficiency analysis (sample/dimension ratios)
- Comprehensive visualization and reporting

This notebook is designed for academic research and provides actionable insights for algorithm selection in high-dimensional scenarios.

## Rules for Claude Code
1. **Follow the Project Structure**: Use the provided directory structure for notebooks, data, models, and scripts.
2. **Use Docker Commands**: When setting up or modifying the environment, use the provided Docker commands
3. **Use English for labels on plots and outputs**: Ensure all visualizations and outputs are in English for consistency.