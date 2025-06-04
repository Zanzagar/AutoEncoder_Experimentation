#!/usr/bin/env python3
"""
Test script to verify latent analysis imports and basic functionality.
"""

def test_latent_analysis_imports():
    """Test that all latent analysis functions can be imported correctly."""
    print("🧪 Testing latent analysis imports...")
    
    try:
        # Test core latent analysis imports
        from autoencoder_lib.experiment.latent_analysis import (
            analyze_latent_space,
            create_latent_tsne_analysis,
            perform_latent_clustering,
            generate_latent_interpolations,
            analyze_latent_traversals,
            calculate_latent_metrics,
            run_complete_latent_analysis
        )
        print("   ✅ Core latent analysis functions imported successfully")
        
        # Test experiment wrapper imports
        from autoencoder_lib.experiment import (
            run_latent_analysis_experiment,
            run_systematic_latent_analysis
        )
        print("   ✅ Experiment wrapper functions imported successfully")
        
        # Test that functions are properly exported from experiment module
        from autoencoder_lib.experiment import (
            analyze_latent_space as exp_analyze_latent_space,
            create_latent_tsne_analysis as exp_create_latent_tsne_analysis,
            perform_latent_clustering as exp_perform_latent_clustering,
            generate_latent_interpolations as exp_generate_latent_interpolations,
            analyze_latent_traversals as exp_analyze_latent_traversals,
            calculate_latent_metrics as exp_calculate_latent_metrics,
            run_complete_latent_analysis as exp_run_complete_latent_analysis
        )
        print("   ✅ All latent analysis functions exported from experiment module")
        
        # Test visualization imports (dependencies)
        from autoencoder_lib.visualization import (
            visualize_latent_space_2d,
            compare_latent_spaces,
            plot_latent_distribution,
            plot_latent_interpolation,
            analyze_latent_clustering,
            plot_latent_variance_analysis,
            visualize_side_by_side_latent_spaces
        )
        print("   ✅ Visualization dependencies imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def test_function_signatures():
    """Test that functions have the expected signatures."""
    print("\n🔍 Testing function signatures...")
    
    try:
        from autoencoder_lib.experiment.latent_analysis import run_complete_latent_analysis
        import inspect
        
        sig = inspect.signature(run_complete_latent_analysis)
        params = list(sig.parameters.keys())
        
        expected_params = [
            'model', 'train_data', 'train_labels', 'test_data', 'test_labels',
            'class_names', 'device', 'output_dir', 'include_interpolations', 'include_traversals'
        ]
        
        for param in expected_params:
            if param in params:
                print(f"   ✅ Parameter '{param}' found")
            else:
                print(f"   ❌ Parameter '{param}' missing")
                return False
        
        print("   ✅ All expected parameters found in run_complete_latent_analysis")
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing function signatures: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting latent analysis import tests...")
    print("=" * 60)
    
    imports_ok = test_latent_analysis_imports()
    signatures_ok = test_function_signatures()
    
    print("\n" + "=" * 60)
    if imports_ok and signatures_ok:
        print("🎉 All tests passed! Latent analysis implementation is ready.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print("=" * 60) 