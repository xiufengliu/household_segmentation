"""
Comprehensive example demonstrating the refactored household energy segmentation system.

This script shows how to use the new modular architecture for:
1. Loading and preprocessing data
2. Training different clustering models
3. Comprehensive evaluation and comparison
4. Statistical significance testing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path

# Import the refactored modules
from src.data.loaders import EnergyDataLoader, WeatherDataLoader
from src.models.autoencoder import ConvAutoencoder, WeatherFusedAutoencoder
from src.clustering.deep_clustering import DeepEmbeddedClustering, WeatherFusedDEC
from src.evaluation.evaluator import ClusteringEvaluator
from src.utils.config import load_config
from src.utils.logging import setup_logging, get_logger
from src.utils.helpers import set_random_seed


def main():
    """Main function demonstrating the comprehensive workflow."""
    
    # Setup logging
    setup_logging(level='INFO', log_file='logs/comprehensive_example.log')
    logger = get_logger(__name__)
    
    logger.info("Starting comprehensive household energy segmentation example")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Load configuration
    config = load_config("configs/default.yaml")
    logger.info(f"Using configuration: {config}")
    
    # Step 1: Load and preprocess data
    logger.info("Step 1: Loading and preprocessing data")
    
    # Load energy consumption data
    energy_loader = EnergyDataLoader(
        expected_timesteps=config.data.n_timesteps,
        expected_features=config.data.n_load_features
    )
    load_data = energy_loader.load_data()
    logger.info(f"Loaded energy data shape: {load_data.shape}")
    
    # Load weather data
    weather_loader = WeatherDataLoader(
        expected_timesteps=config.data.n_timesteps,
        expected_features=config.data.n_weather_features
    )
    weather_data = weather_loader.load_data()
    logger.info(f"Loaded weather data shape: {weather_data.shape}")
    
    # Step 2: Train different clustering models
    logger.info("Step 2: Training clustering models")
    
    # 2a. Traditional Deep Embedded Clustering (DEC)
    logger.info("Training traditional DEC model...")
    
    # Build and train autoencoder
    autoencoder = ConvAutoencoder(
        input_shape=(config.data.n_timesteps, config.data.n_load_features),
        embedding_dim=config.model.embedding_dim
    )
    autoencoder.compile()
    
    # Pre-train autoencoder
    autoencoder.fit(
        load_data,
        epochs=config.training.autoencoder_epochs,
        batch_size=config.model.batch_size,
        verbose=1
    )
    
    # Create DEC model
    dec_model = DeepEmbeddedClustering(
        n_clusters=config.model.n_clusters,
        embedding_dim=config.model.embedding_dim
    )
    
    # Initialize with pre-trained autoencoder
    dec_model.initialize_with_autoencoder(autoencoder)
    
    # Train DEC
    dec_labels = dec_model.fit_predict(
        load_data,
        epochs=config.training.clustering_epochs,
        batch_size=config.model.batch_size
    )
    
    logger.info(f"DEC clustering completed. Cluster distribution: {np.bincount(dec_labels)}")
    
    # 2b. Weather-Fused Deep Embedded Clustering
    logger.info("Training weather-fused DEC model...")
    
    # Build and train weather-fused autoencoder
    weather_autoencoder = WeatherFusedAutoencoder(
        load_shape_input_shape=(config.data.n_timesteps, config.data.n_load_features),
        weather_input_shape=(config.data.n_timesteps, config.data.n_weather_features),
        embedding_dim=config.model.embedding_dim
    )
    weather_autoencoder.compile()
    
    # Pre-train weather-fused autoencoder
    weather_autoencoder.fit(
        load_data, weather_data,
        epochs=config.training.autoencoder_epochs,
        batch_size=config.model.batch_size,
        verbose=1
    )
    
    # Create weather-fused DEC model
    weather_dec_model = WeatherFusedDEC(
        n_clusters=config.model.n_clusters,
        embedding_dim=config.model.embedding_dim
    )
    
    # Initialize with pre-trained weather-fused autoencoder
    weather_dec_model.initialize_with_autoencoder(weather_autoencoder)
    
    # Train weather-fused DEC
    weather_dec_labels = weather_dec_model.fit_predict(
        load_data, weather_data,
        epochs=config.training.clustering_epochs,
        batch_size=config.model.batch_size
    )
    
    logger.info(f"Weather-fused DEC clustering completed. Cluster distribution: {np.bincount(weather_dec_labels)}")
    
    # Step 3: Comprehensive evaluation
    logger.info("Step 3: Comprehensive evaluation")
    
    evaluator = ClusteringEvaluator(
        include_uncertainty=True,
        include_stability=True,
        include_statistical_tests=True
    )
    
    # Evaluate traditional DEC
    logger.info("Evaluating traditional DEC...")
    dec_embeddings = dec_model.get_embeddings(load_data)
    dec_results = evaluator.evaluate(
        X=dec_embeddings,
        labels=dec_labels,
        clustering_method=dec_model,
        method_name="Traditional_DEC"
    )
    
    # Evaluate weather-fused DEC
    logger.info("Evaluating weather-fused DEC...")
    weather_embeddings = weather_dec_model.get_embeddings(load_data, weather_data)
    weather_results = evaluator.evaluate(
        X=weather_embeddings,
        labels=weather_dec_labels,
        clustering_method=weather_dec_model,
        method_name="Weather_Fused_DEC"
    )
    
    # Step 4: Compare methods
    logger.info("Step 4: Comparing clustering methods")
    
    comparison_df = evaluator.compare_methods(
        [dec_results, weather_results],
        primary_metric='silhouette_score'
    )
    
    logger.info("Method comparison results:")
    logger.info(f"\n{comparison_df}")
    
    # Step 5: Statistical significance testing
    logger.info("Step 5: Statistical significance testing")
    
    # Compare the two methods statistically
    from src.evaluation.statistical_tests import StatisticalTester
    
    statistical_tester = StatisticalTester()
    comparison_results = statistical_tester.compare_clustering_methods(
        X=dec_embeddings,  # Use same embedding space for fair comparison
        labels_list=[dec_labels, weather_dec_labels],
        method_names=["Traditional_DEC", "Weather_Fused_DEC"]
    )
    
    logger.info("Statistical comparison results:")
    for key, value in comparison_results.items():
        logger.info(f"{key}: {value}")
    
    # Step 6: Generate comprehensive reports
    logger.info("Step 6: Generating reports")
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Generate individual reports
    dec_report = evaluator.generate_report(
        "Traditional_DEC",
        output_path=output_dir / "traditional_dec_report.txt"
    )
    
    weather_report = evaluator.generate_report(
        "Weather_Fused_DEC", 
        output_path=output_dir / "weather_fused_dec_report.txt"
    )
    
    # Save all results
    evaluator.save_results(output_dir / "all_evaluation_results.json")
    
    # Save comparison results
    comparison_df.to_csv(output_dir / "method_comparison.csv", index=False)
    
    # Step 7: Identify best method
    logger.info("Step 7: Identifying best performing method")
    
    best_method, best_score = evaluator.get_best_method(
        metric='silhouette_score',
        higher_is_better=True
    )
    
    logger.info(f"Best performing method: {best_method} (Silhouette Score: {best_score:.4f})")
    
    # Step 8: Novelty analysis and recommendations
    logger.info("Step 8: Novelty analysis for TNNLS submission")
    
    novelty_analysis = analyze_novelty(dec_results, weather_results, comparison_results)
    
    logger.info("Novelty analysis:")
    for point in novelty_analysis:
        logger.info(f"- {point}")
    
    logger.info("Comprehensive example completed successfully!")
    
    return {
        'dec_results': dec_results,
        'weather_results': weather_results,
        'comparison_df': comparison_df,
        'statistical_comparison': comparison_results,
        'best_method': best_method,
        'best_score': best_score,
        'novelty_analysis': novelty_analysis
    }


def analyze_novelty(dec_results, weather_results, statistical_comparison):
    """
    Analyze the novelty of the weather-fused approach for TNNLS submission.
    
    Args:
        dec_results: Results from traditional DEC
        weather_results: Results from weather-fused DEC
        statistical_comparison: Statistical comparison results
        
    Returns:
        List of novelty points for the paper
    """
    novelty_points = []
    
    # Performance improvement analysis
    dec_silhouette = dec_results['internal_metrics']['silhouette_score']
    weather_silhouette = weather_results['internal_metrics']['silhouette_score']
    
    if weather_silhouette > dec_silhouette:
        improvement = ((weather_silhouette - dec_silhouette) / dec_silhouette) * 100
        novelty_points.append(
            f"Weather fusion improves silhouette score by {improvement:.2f}% "
            f"({dec_silhouette:.4f} → {weather_silhouette:.4f})"
        )
    
    # Statistical significance
    pairwise_key = list(statistical_comparison['pairwise_comparisons'].keys())[0]
    pairwise_result = statistical_comparison['pairwise_comparisons'][pairwise_key]
    
    if pairwise_result.get('significant', False):
        novelty_points.append(
            f"Improvement is statistically significant (p-value: {pairwise_result['p_value']:.4f})"
        )
    
    # Uncertainty quantification
    if weather_results['uncertainty_metrics']:
        uncertainty = weather_results['uncertainty_metrics']
        novelty_points.append(
            f"Weather-fused model provides uncertainty quantification with "
            f"{uncertainty['high_confidence_proportion']:.2%} high-confidence assignments"
        )
    
    # Multi-modal fusion
    novelty_points.append(
        "Novel multi-modal fusion architecture combining load patterns with weather data"
    )
    
    # Attention mechanism
    novelty_points.append(
        "Cross-modal attention mechanism for interpretable weather-load relationships"
    )
    
    # Practical implications
    novelty_points.append(
        "Framework enables weather-aware demand response and energy management strategies"
    )
    
    return novelty_points


if __name__ == "__main__":
    results = main()
