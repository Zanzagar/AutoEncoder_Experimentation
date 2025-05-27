# AutoEncoder Experimentation Framework

A comprehensive Python package for systematic exploration of autoencoder latent space discrimination capabilities across different datasets and architectures.

## 🎯 Project Overview

This project investigates how autoencoder neural networks represent different types of images and geometric features in latent space. The primary research goals are to:

- Understand latent space discrimination capabilities
- Determine when latent space breaks down for specific features
- Identify minimum latent features needed for effective discrimination
- Analyze how incrementally complex features are encoded

## 🚀 Current Status

**Phase 0: Migration from Jupyter Notebook to Python Package**

- ✅ TaskMaster-AI project management initialized
- ✅ Git repository setup with comprehensive .gitignore
- 🔄 Migrating from `AutoEncoderJupyterTest.ipynb` to modular Python package
- 🔄 Creating four core wrapper functions for clean notebook interface

## 📁 Project Structure

```
AutoEncoder_Experimentation/
├── autoencoder_lib/          # Target Python package (to be created)
│   ├── data/                 # Dataset generation and loading utilities
│   ├── models/               # Autoencoder architecture definitions
│   ├── experiment/           # Training and experiment management
│   ├── visualization/        # Visualization functions and utilities
│   └── utils/                # Helper functions and utilities
├── scripts/                  # Project documentation and configuration
│   ├── prd.txt              # Product Requirements Document
│   └── task-complexity-report.json
├── tasks/                    # TaskMaster-AI task management
├── AutoEncoderJupyterTest.ipynb    # Original implementation (reference)
└── README.md                 # This file
```

## 🔬 Research Datasets

### Geological Datasets
- **Consistent Layers**: Layered geological patterns with consistent properties
- **Variable Layers**: Layered patterns with variable characteristics
- Designed to test discrimination of subtle geological features

### Progressive Complexity
- Starting with similar classes containing extremely subtle differences
- Gradually adding more complex features to probe latent space thresholds
- Framework designed to handle any generic dataset

## 🏗️ Target Architecture

### Four Core Wrapper Functions
1. **Dataset Generation Wrapper**: Generate any generic dataset with visualization
2. **Dataset Visualization Wrapper**: t-SNE projection and dataset analysis
3. **Experiment Runner Wrapper**: Systematic exploration of autoencoder models
4. **Results Loader Wrapper**: Load and visualize results from saved experiments

### Autoencoder Architectures
- Standard symmetric autoencoders
- Variational autoencoders (VAEs)
- Convolutional autoencoders
- Adversarial autoencoders
- Sparse autoencoders

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy, Matplotlib, Scikit-learn
- Jupyter Notebook

### Installation (Coming Soon)
```bash
# Clone the repository
git clone https://github.com/yourusername/AutoEncoder_Experimentation.git
cd AutoEncoder_Experimentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Current Workflow (Jupyter Notebook)

The existing workflow consists of 5 sequential code blocks:

1. **Dataset Generation**: Generates datasets with visualization and class examples
2. **Dataset Visualization**: t-SNE projection visualization of specified dataset
3. **Model Definitions**: Defines autoencoder architectures and model classes
4. **Experiment Execution**: Systematic exploration with specified parameters
5. **Results Loading**: Retrieves and visualizes results from saved experiments

## 🎯 Migration Goals

### Critical Requirements
- ✅ Preserve 100% of current functionality
- 🔄 Fix visualization consistency issues between run and load operations
- 🔄 Support any generic dataset and autoencoder architecture
- 🔄 Implement high-level parameter configuration
- 🔄 Create clean, maintainable Python package structure

### Key Improvements
- **Reproducibility**: Consistent visualizations with explicit seed management
- **Generalization**: Abstract interfaces for datasets and architectures
- **Maintainability**: Modular code structure for future development
- **Usability**: Simple, intuitive interface for complex experiments

## 📈 Task Management

This project uses [TaskMaster-AI](https://github.com/taskmaster-ai/taskmaster-ai) for systematic task management:

- **15 main tasks** organized by priority and dependencies
- **AI-powered complexity analysis** for task breakdown
- **Research-backed task generation** using Perplexity AI
- **Comprehensive progress tracking** and dependency management

View current tasks: `task-master list`
Get next task: `task-master next`

## 🤝 Contributing

This project follows a strict Git workflow:

1. **Feature Branches**: All development on feature branches
2. **Descriptive Commits**: Clear, actionable commit messages
3. **Pull Requests**: All merges via pull request review
4. **No Direct Main Commits**: Main branch protected

### Branch Naming Convention
- `feature/migrate-data-module`
- `feature/migrate-models-module`
- `bugfix/visualization-consistency`
- `docs/update-documentation`

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Related Work

- Original Jupyter notebook implementation: `AutoEncoderJupyterTest.ipynb`
- Backup implementation: `AutoEncoderJupyterTest_BACKUP.ipynb`
- Project documentation: `scripts/prd.txt`

## 📞 Contact

For questions about this research project, please open an issue or contact the development team.

---

**Status**: 🚧 Active Development - Migration Phase 0
**Last Updated**: 2025-05-27 