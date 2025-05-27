---
description: 
globs: 
alwaysApply: true
---
# AutoEncoder Experimentation Project Overview

This project investigates how autoencoder neural networks represent different types of images and geometric features in the latent space. The primary goal is to understand latent space discrimination capabilities, determine when the latent space breaks down, and identify the minimum set of latent features needed to discriminate between different classes of images.

## Current Status

- Working autoencoder implementation exists in `AutoEncoderJupyterTest.ipynb` with 5 sequential code blocks
- Initial experiments with geological datasets have been conducted
- Several autoencoder architectures have been tested with different latent dimensions
- Results are stored in `latent_dim_exploration_results_layered_geology/`
- **Ready for migration**: Current notebook workflow is functional but needs to be reorganized into a Python package
- **Git repository**: Project should be under version control with proper Git workflow

## Current Workflow (Jupyter Notebook)

The existing workflow consists of 5 sequential code blocks:

1. **Code Block 1 - Dataset Generation**: Generates datasets with visualization and examples for each class
2. **Code Block 2 - Dataset Visualization**: t-SNE projection visualization of dataset at specified input path
3. **Code Block 3 - Model Definitions**: Defines autoencoder architectures, model classes, and dataset handling (no output)
4. **Code Block 4 - Experiment Execution**: Systematic exploration of autoencoder models with specified parameters
5. **Code Block 5 - Results Loading**: Retrieves and visualizes results from previously saved experiments

## Immediate Priority: Migration to Python Package

### Git Workflow Integration

**Version Control Strategy**:
- All migration work follows feature branch workflow
- Never commit directly to main branch
- Use descriptive branch names: `feature/migrate-data-module`, `bugfix/visualization-consistency`
- Commit frequently with descriptive messages
- Push for backup at end of each work day
- Create pull requests for all merges to main

**Daily Development Workflow**:
1. Start: `git checkout main && git pull origin main && git checkout -b feature/task-name`
2. Work: Make changes, test functionality
3. Save: `git add . && git commit -m "Descriptive message"`
4. Backup: `git push origin feature/task-name`
5. Merge: Create pull request for code review

### Target Workflow (New Python Package + Jupyter Interface)

**Package Structure**: `autoencoder_lib/`
- `data/`: Dataset generation and loading utilities
- `models/`: Autoencoder architecture definitions and model classes  
- `experiment/`: Training, evaluation, and experiment management
- `visualization/`: All visualization functions and utilities
- `utils/`: Helper functions and utilities

**New Jupyter Interface**: Four core wrapper functions
1. **Dataset Generation Wrapper**: Generate any generic dataset with visualization
2. **Dataset Visualization Wrapper**: t-SNE projection and dataset analysis  
3. **Experiment Runner Wrapper**: Systematic exploration of autoencoder models
4. **Results Loader Wrapper**: Load and visualize results from saved experiments

### Critical Issues to Resolve

**Visualization Consistency Problems**:
- t-SNE projections use different seeds between run and load operations
- Training/test reconstruction visualizations show different images (should be same 2 per class)
- Performance grid missing training data (currently only shows test data)
- Run vs load visualizations are inconsistent and need to be identical

**Generalization Requirements**:
- Support any generic dataset input
- Support any generic autoencoder architecture
- High-level parameter configuration (learning rate, architectures, random seed, latent dimensions)
- Maintain ALL current visualization and functionality

## Project Structure

**Current Files**:
- `AutoEncoderJupyterTest.ipynb`: Main notebook (source for migration)
- `AutoEncoderJupyterTest_BACKUP.ipynb`: Backup of the main notebook
- `autoencoder_lib/`: Target directory for Python package (to be created)

**Target Structure**:
- `autoencoder_lib/`: Python package with modular components
- `AutoEncoderWrapper.ipynb`: New notebook interface using the package
- Existing notebooks preserved for reference

**Git Repository Structure**:
- `main` branch: Stable, working code only
- Feature branches: `feature/module-name` for each migration task
- Bug fix branches: `bugfix/issue-description` for specific fixes
- Documentation branches: `docs/update-type` for documentation updates

## Datasets

The project uses progressively complex datasets:
- Starting with similar classes containing extremely subtle differences
- Gradually adding more complex features to probe latent space discrimination thresholds
- Current geological datasets:
  - `layered_geologic_patterns_dataset/consistent_layers/`
  - `layered_geologic_patterns_dataset/variable_layers/`
- Framework designed to handle any generic dataset

## Research Goals

- Explore how different autoencoder architectures represent features in latent space
- Identify when latent space discrimination breaks down for specific features and image types
- Determine the minimum set of latent features required for effective discrimination
- Analyze how incrementally complex features are encoded in the latent space

## Development Philosophy

**No Jupyter Notebook Development**: All future development and debugging will occur in Python files within the `autoencoder_lib` package. The Jupyter notebook will serve only as a clean interface for running the four core wrapper functions.

**Preserve All Functionality**: The migration must retain 100% of current visualization and analytical capabilities while making the framework more general and maintainable.

**Reproducibility Focus**: All visualization consistency issues must be resolved to ensure that run and load operations produce identical results.

**Version Control Discipline**: All code changes must be tracked in Git with descriptive commits. No direct commits to main branch. All merges via pull requests.

## Git Workflow Rules for Migration

### Branch Naming Conventions
- `feature/migrate-data-module`: Data module migration
- `feature/migrate-models-module`: Models module migration
- `feature/migrate-experiment-module`: Experiment module migration
- `feature/migrate-visualization-module`: Visualization module migration
- `feature/create-wrapper-functions`: Four core wrapper functions
- `bugfix/visualization-consistency`: Fix visualization consistency issues
- `bugfix/seed-consistency`: Fix random seed consistency
- `bugfix/image-selection-consistency`: Fix image selection consistency
- `docs/update-documentation`: Documentation updates

### Commit Guidelines
**Good Commit Messages**:
- "Migrate dataset generation functions from notebook to data module"
- "Fix t-SNE seed consistency between run and load operations"
- "Add training data to performance grid visualization"
- "Create wrapper function for experiment execution"

**When to Commit**:
- ✅ Module migration completed and tested
- ✅ Bug fixed and verified
- ✅ Wrapper function working correctly
- ✅ Before risky refactoring
- ✅ End of work session

**When NOT to Commit**:
- ❌ Code doesn't run or has errors
- ❌ Broken functionality
- ❌ Incomplete module migration

### Emergency Recovery Procedures
- **Uncommitted changes**: `git stash` or `git checkout -- filename.py`
- **Committed but not pushed**: `git reset HEAD~1`
- **Pushed changes**: `git revert HEAD && git push origin main`
- **Lost work**: `git reflog` to find and recover

## Results

Experimental results are stored in:
- `latent_dim_exploration_results_layered_geology/`: Results from exploring different latent dimensions
- Additional result directories will be created for each dataset/architecture combination
- Results loading and visualization must be consistent with original experiment execution
- All result generation and analysis code tracked in version control





