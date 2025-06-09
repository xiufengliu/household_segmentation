#!/usr/bin/env python3
"""
Generate synthetic household energy consumption and weather data.

This script uses the SyntheticDataGenerator to create realistic synthetic
datasets for testing and development of the clustering algorithms.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('src')

# Import only what we need to avoid TensorFlow dependencies
from data.synthetic import SyntheticDataGenerator


def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    """Generate synthetic data and save as CSV files."""

    print("Starting synthetic data generation...")

    # Create output directory
    output_dir = Path("data/synthetic")
    ensure_dir(output_dir)
    
    # Initialize generator
    generator = SyntheticDataGenerator(random_state=42)
    
    # Generation parameters
    n_samples = 500  # Number of households
    n_timesteps = 24  # 24 hours
    n_clusters = 5   # Number of clusters
    
    print(f"Generating {n_samples} samples with {n_clusters} clusters")

    # Generate clustered data with known ground truth
    load_shapes, weather_data, true_labels = generator.generate_clustered_data(
        n_samples=n_samples,
        n_clusters=n_clusters,
        n_timesteps=n_timesteps,
        cluster_separation=0.6
    )

    print(f"Generated data shapes:")
    print(f"  Load shapes: {load_shapes.shape}")
    print(f"  Weather data: {weather_data.shape}")
    print(f"  True labels: {true_labels.shape}")

    # Save load shapes as CSV
    print("Saving load shapes data...")
    load_df = pd.DataFrame()
    
    # Add household ID
    load_df['household_id'] = range(n_samples)
    
    # Add load data for each hour
    for hour in range(n_timesteps):
        load_df[f'load_hour_{hour:02d}'] = load_shapes[:, hour, 0]
    
    # Add cluster labels
    load_df['true_cluster'] = true_labels
    
    # Save to CSV
    load_csv_path = output_dir / "load_shapes.csv"
    load_df.to_csv(load_csv_path, index=False)
    print(f"Load shapes saved to: {load_csv_path}")

    # Save weather data as CSV
    print("Saving weather data...")
    weather_df = pd.DataFrame()

    # Add household ID
    weather_df['household_id'] = range(n_samples)

    # Add temperature data for each hour
    for hour in range(n_timesteps):
        weather_df[f'temperature_hour_{hour:02d}'] = weather_data[:, hour, 0]

    # Add humidity data for each hour
    for hour in range(n_timesteps):
        weather_df[f'humidity_hour_{hour:02d}'] = weather_data[:, hour, 1]

    # Add cluster labels
    weather_df['true_cluster'] = true_labels

    # Save to CSV
    weather_csv_path = output_dir / "weather_data.csv"
    weather_df.to_csv(weather_csv_path, index=False)
    print(f"Weather data saved to: {weather_csv_path}")

    # Create a combined dataset
    print("Creating combined dataset...")
    combined_df = pd.DataFrame()
    
    # Add household ID
    combined_df['household_id'] = range(n_samples)
    
    # Add load and weather data side by side
    for hour in range(n_timesteps):
        combined_df[f'load_h{hour:02d}'] = load_shapes[:, hour, 0]
        combined_df[f'temp_h{hour:02d}'] = weather_data[:, hour, 0]
        combined_df[f'humid_h{hour:02d}'] = weather_data[:, hour, 1]
    
    # Add cluster labels
    combined_df['true_cluster'] = true_labels
    
    # Save combined dataset
    combined_csv_path = output_dir / "combined_data.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined data saved to: {combined_csv_path}")

    # Generate additional datasets for different scenarios
    print("Generating additional datasets...")
    
    # 1. Small dataset for quick testing
    small_load, small_weather, small_labels = generator.generate_clustered_data(
        n_samples=100,
        n_clusters=3,
        n_timesteps=24,
        cluster_separation=0.8
    )
    
    # Save small dataset
    small_df = pd.DataFrame()
    small_df['household_id'] = range(100)
    
    for hour in range(24):
        small_df[f'load_h{hour:02d}'] = small_load[:, hour, 0]
        small_df[f'temp_h{hour:02d}'] = small_weather[:, hour, 0]
        small_df[f'humid_h{hour:02d}'] = small_weather[:, hour, 1]
    
    small_df['true_cluster'] = small_labels
    
    small_csv_path = output_dir / "small_dataset.csv"
    small_df.to_csv(small_csv_path, index=False)
    print(f"Small dataset saved to: {small_csv_path}")

    # 2. Large dataset for comprehensive testing
    large_load, large_weather, large_labels = generator.generate_clustered_data(
        n_samples=1000,
        n_clusters=8,
        n_timesteps=24,
        cluster_separation=0.4
    )

    # Save large dataset (load shapes only for size)
    large_load_df = pd.DataFrame()
    large_load_df['household_id'] = range(1000)

    for hour in range(24):
        large_load_df[f'load_hour_{hour:02d}'] = large_load[:, hour, 0]

    large_load_df['true_cluster'] = large_labels

    large_csv_path = output_dir / "large_load_dataset.csv"
    large_load_df.to_csv(large_csv_path, index=False)
    print(f"Large dataset saved to: {large_csv_path}")

    # 3. Generate correlated data
    print("Generating correlated load-weather data...")
    corr_load, corr_weather = generator.generate_correlated_data(
        n_samples=300,
        n_timesteps=24,
        correlation_strength=0.5
    )
    
    # Save correlated dataset
    corr_df = pd.DataFrame()
    corr_df['household_id'] = range(300)
    
    for hour in range(24):
        corr_df[f'load_h{hour:02d}'] = corr_load[:, hour, 0]
        corr_df[f'temp_h{hour:02d}'] = corr_weather[:, hour, 0]
        corr_df[f'humid_h{hour:02d}'] = corr_weather[:, hour, 1]
    
    corr_csv_path = output_dir / "correlated_data.csv"
    corr_df.to_csv(corr_csv_path, index=False)
    print(f"Correlated data saved to: {corr_csv_path}")

    # Generate summary statistics
    print("Generating summary statistics...")
    
    summary_stats = {
        'dataset': ['main', 'small', 'large', 'correlated'],
        'n_samples': [n_samples, 100, 1000, 300],
        'n_clusters': [n_clusters, 3, 8, 'N/A'],
        'n_timesteps': [24, 24, 24, 24],
        'load_mean': [
            np.mean(load_shapes),
            np.mean(small_load),
            np.mean(large_load),
            np.mean(corr_load)
        ],
        'load_std': [
            np.std(load_shapes),
            np.std(small_load),
            np.std(large_load),
            np.std(corr_load)
        ],
        'weather_temp_mean': [
            np.mean(weather_data[:, :, 0]),
            np.mean(small_weather[:, :, 0]),
            np.mean(large_weather[:, :, 0]),
            np.mean(corr_weather[:, :, 0])
        ],
        'weather_humid_mean': [
            np.mean(weather_data[:, :, 1]),
            np.mean(small_weather[:, :, 1]),
            np.mean(large_weather[:, :, 1]),
            np.mean(corr_weather[:, :, 1])
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_csv_path = output_dir / "dataset_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary statistics saved to: {summary_csv_path}")

    # Create metadata file
    metadata = {
        'generation_date': pd.Timestamp.now().isoformat(),
        'generator_version': '1.0',
        'random_seed': 42,
        'datasets_created': [
            'load_shapes.csv',
            'weather_data.csv', 
            'combined_data.csv',
            'small_dataset.csv',
            'large_load_dataset.csv',
            'correlated_data.csv',
            'dataset_summary.csv'
        ],
        'data_description': {
            'load_shapes': 'Household energy consumption patterns (normalized)',
            'weather_data': 'Temperature and humidity patterns (normalized)',
            'combined_data': 'Load and weather data combined',
            'small_dataset': 'Small dataset for quick testing (100 samples, 3 clusters)',
            'large_load_dataset': 'Large dataset for comprehensive testing (1000 samples, 8 clusters)',
            'correlated_data': 'Load-weather correlated data (correlation_strength=0.5)',
            'dataset_summary': 'Statistical summary of all datasets'
        },
        'column_descriptions': {
            'household_id': 'Unique identifier for each household',
            'load_h##': 'Energy consumption at hour ## (0-23)',
            'temp_h##': 'Temperature at hour ## (normalized)',
            'humid_h##': 'Humidity at hour ## (normalized)',
            'true_cluster': 'Ground truth cluster assignment'
        }
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_csv_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)
    print(f"Metadata saved to: {metadata_csv_path}")

    print("Synthetic data generation completed successfully!")
    print(f"All files saved to: {output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("SYNTHETIC DATA GENERATION SUMMARY")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Total datasets created: {len(metadata['datasets_created'])}")
    print("\nDatasets:")
    for i, dataset in enumerate(metadata['datasets_created'], 1):
        print(f"  {i}. {dataset}")
    print("\nMain dataset statistics:")
    print(f"  Samples: {n_samples}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Load mean: {np.mean(load_shapes):.4f}")
    print(f"  Load std: {np.std(load_shapes):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
