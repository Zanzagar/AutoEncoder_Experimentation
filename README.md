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
- ✅ GitHub repository created and connected
- ✅ Development environment configured
- ✅ Python package structure created
- 🔄 Data module implementation (in progress)

## 📓 Jupyter Notebook Development

### **CRITICAL: notebook_mcp Tools Required**

**⚠️ IMPORTANT**: All Jupyter notebook operations MUST use the `notebook_mcp` tools to prevent corruption:

- **NEVER use standard file editing tools** on `.ipynb` files
- **ALWAYS use notebook_mcp tools** for reading, editing, and creating notebooks
- **Investigate notebook issues independently** using available MCP tools

### Available Tools
- **File Operations**: Create, delete, rename notebooks
- **Cell Operations**: Add, edit, delete, move, split, merge cells
- **Content Reading**: Read cells, metadata, outputs
- **Validation**: Validate notebook structure
- **Export**: Convert to other formats

### Best Practices
- Use notebooks as **interfaces** to the `autoencoder_lib` package
- Keep **core logic in Python modules**, not notebook cells
- **Import and use package functions** rather than defining in notebooks
- Maintain **clean separation** between package code and notebook interface

See [`.cursor/rules/jupyter-notebook-mcp.mdc`](.cursor/rules/jupyter-notebook-mcp.mdc) for complete guidelines.

## 🛠️ Development Environment Setup

### Prerequisites
- Python 3.9+ (Anaconda recommended)
- Git
- GitHub account

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Zanzagar/AutoEncoder_Experimentation.git
   cd AutoEncoder_Experimentation
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch, numpy, matplotlib; print('Environment setup successful!')"
   ```

### Core Dependencies

- **Machine Learning**: PyTorch, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data Processing**: Pandas, H5PY
- **Development**: Jupyter, Black, Flake8, Pytest
- **Dimensionality Reduction**: UMAP, t-SNE

## 📁 Project Structure

```
AutoEncoder_Experimentation/
├── autoencoder_lib/          # Python package (to be created)
│   ├── data/                 # Dataset generation and loading
│   ├── models/               # Autoencoder architectures
│   ├── experiment/           # Training and evaluation
│   ├── visualization/        # Plotting and analysis
│   └── utils/                # Helper functions
├── notebooks/                # Jupyter interface notebooks
├── scripts/                  # Utility scripts and PRD
├── tasks/                    # TaskMaster-AI task management
├── results/                  # Experiment outputs
└── tests/                    # Unit tests
```

## 🔬 Research Datasets

### Current Datasets
- **Geological Patterns**: Layered geological structures with varying complexity
  - `consistent_layers/`: Regular geological patterns
  - `variable_layers/`: Irregular geological patterns

### Dataset Framework
The framework is designed to handle any generic dataset with the following capabilities:
- Automatic dataset generation with visualization
- t-SNE projection analysis
- Class-based example visualization
- Configurable complexity levels

## 🧠 Autoencoder Architectures

### Supported Architectures
- **BasicAutoEncoder**: Simple encoder-decoder structure
- **ConvolutionalAutoEncoder**: CNN-based architecture for image data
- **VariationalAutoEncoder**: VAE for probabilistic latent representations
- **DeepAutoEncoder**: Multi-layer deep architecture

### Architecture Framework
- Generic architecture interface
- Configurable latent dimensions
- Modular encoder/decoder components
- Support for custom architectures

## 🎮 Current Workflow (Jupyter Notebook)

The existing workflow consists of 5 sequential code blocks:

1. **Dataset Generation**: Creates datasets with visualization and class examples
2. **Dataset Visualization**: t-SNE projection of input datasets
3. **Model Definitions**: Autoencoder architectures and dataset handling
4. **Experiment Execution**: Systematic exploration with specified parameters
5. **Results Loading**: Retrieval and visualization of saved experiments

## 🎯 Target Workflow (Python Package + Jupyter Interface)

### Four Core Wrapper Functions
1. **Dataset Generation Wrapper**: Generate any generic dataset with visualization
2. **Dataset Visualization Wrapper**: t-SNE projection and dataset analysis
3. **Experiment Runner Wrapper**: Systematic exploration of autoencoder models
4. **Results Loader Wrapper**: Load and visualize results from saved experiments

## 📊 Experiment Configuration

### High-Level Parameters
- **Learning Rate**: Optimizer learning rate
- **Architectures**: List of autoencoder architectures to test
- **Random Seed**: Consistent reproducibility
- **Latent Dimensions**: Range of latent space dimensions
- **Dataset Path**: Input dataset location
- **Output Directory**: Results storage location

## 🔍 Visualization Features

### Current Visualizations
- Dataset class examples
- t-SNE projections
- Training/validation loss curves
- Reconstruction quality comparisons
- Latent space analysis
- Performance grids

### Consistency Requirements
- Identical visualizations between run and load operations
- Consistent random seeds for reproducibility
- Same image selection for reconstruction examples
- Unified color schemes and layouts

## 🚧 Known Issues

### Visualization Consistency Problems
- t-SNE projections use different seeds between run and load operations
- Training/test reconstruction visualizations show different images
- Performance grid missing training data (currently only shows test data)
- Run vs load visualizations are inconsistent

## 🤝 Contributing

### Git Workflow
- **Never commit directly to main branch**
- Use feature branches: `feature/module-name`
- Create pull requests for all merges
- Follow commit message conventions

### Branch Naming Conventions
- `feature/migrate-data-module`: Data module migration
- `feature/migrate-models-module`: Models module migration
- `feature/migrate-experiment-module`: Experiment module migration
- `feature/migrate-visualization-module`: Visualization module migration
- `bugfix/visualization-consistency`: Fix visualization issues

### Development Rules
- All development in Python files within `autoencoder_lib` package
- Jupyter notebooks serve only as clean interfaces
- Preserve 100% of current functionality
- Fix visualization consistency issues
- Maintain reproducibility

## 📈 TaskMaster-AI Integration

This project uses TaskMaster-AI for project management:
- Comprehensive task breakdown and tracking
- AI-powered task complexity analysis
- Dependency management
- Progress monitoring

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- **Repository**: https://github.com/Zanzagar/AutoEncoder_Experimentation
- **Issues**: https://github.com/Zanzagar/AutoEncoder_Experimentation/issues
- **TaskMaster-AI**: Integrated project management system

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

## 📞 Contact

For questions about this research project, please open an issue or contact the development team.

---

**Status**: 🚧 Active Development - Migration Phase 0
**Last Updated**: 2025-05-27 