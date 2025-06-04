#!/usr/bin/env python3
"""
Test script to verify Optuna integration imports and basic functionality.
"""

def test_optuna_imports():
    """Test that all Optuna integration functions can be imported correctly."""
    print("🧪 Testing Optuna integration imports...")
    
    try:
        # Test core Optuna optimization imports
        from autoencoder_lib.experiment.optuna_optimization import (
            OptunaObjective,
            define_hyperparameter_search_space,
            create_optuna_study,
            run_optuna_optimization,
            analyze_optuna_results,
            run_optuna_systematic_optimization
        )
        print("   ✅ Core Optuna optimization functions imported successfully")
        
        # Test experiment wrapper imports
        from autoencoder_lib.experiment.wrappers import (
            run_optuna_experiment_optimization,
            run_multi_metric_optuna_optimization,
            create_optuna_configuration_from_experiment
        )
        print("   ✅ Optuna experiment wrapper functions imported successfully")
        
        # Test experiment module integration
        from autoencoder_lib.experiment import (
            OptunaObjective,
            define_hyperparameter_search_space,
            create_optuna_study,
            run_optuna_optimization,
            analyze_optuna_results,
            run_optuna_systematic_optimization,
            run_optuna_experiment_optimization,
            run_multi_metric_optuna_optimization,
            create_optuna_configuration_from_experiment
        )
        print("   ✅ All Optuna functions accessible from experiment module")
        
        # Test Optuna library import
        import optuna
        print("   ✅ Optuna library imported successfully")
        
        # Test basic configuration creation
        search_space = define_hyperparameter_search_space()
        print(f"   ✅ Search space creation works: {len(search_space)} parameters")
        
        # Test configuration helper
        dummy_experiment_config = {
            'epochs': 10,
            'architectures': ['simple_linear']
        }
        optuna_config = create_optuna_configuration_from_experiment(dummy_experiment_config)
        print(f"   ✅ Configuration helper works: {optuna_config['metric']}")
        
        print("\n🎉 All Optuna integration tests passed!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def test_optuna_objective_creation():
    """Test that OptunaObjective can be created correctly."""
    print("\n🧪 Testing OptunaObjective creation...")
    
    try:
        from autoencoder_lib.experiment.optuna_optimization import OptunaObjective
        
        # Create test configuration
        dataset_config = {
            'dataset_type': 'layered_geology',
            'n_samples_per_class': 10,
            'image_size': (64, 64)
        }
        
        architecture_names = ['simple_linear']
        fixed_params = {'epochs': 5}
        
        # Create objective
        objective = OptunaObjective(
            dataset_config=dataset_config,
            architecture_names=architecture_names,
            fixed_params=fixed_params,
            verbose=False
        )
        
        print("   ✅ OptunaObjective created successfully")
        print(f"   📊 Optimization metric: {objective.optimization_metric}")
        print(f"   🏗️ Architectures: {objective.architecture_names}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating OptunaObjective: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Testing Optuna Integration for Task 10.7")
    print("=" * 60)
    
    # Run tests
    test1_result = test_optuna_imports()
    test2_result = test_optuna_objective_creation()
    
    print("\n" + "=" * 60)
    if test1_result and test2_result:
        print("🏆 All Optuna integration tests PASSED!")
        print("✅ Task 10.7 implementation ready for use")
    else:
        print("❌ Some tests FAILED - needs debugging")
    
    print("=" * 60) 