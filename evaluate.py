import argparse
import os
from pathlib import Path

import torch
import yaml

from data_loaders import get_data_loaders
from metrics import MetricsCalculator, print_metrics, save_metrics
from model import MosFPAD
from transforms import get_transforms


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_checkpoint(model, checkpoint_path):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint loaded from {checkpoint_path}")

    # Print epoch info if available
    if "epoch" in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")
    if "best_acc" in checkpoint:
        print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")

    return model


def evaluate(model, test_loader, device):
    """
    Evaluate model on test set.

    Args:
        model: The trained model
        test_loader: DataLoader for test set
        device: torch device

    Returns:
        dict: Test metrics
    """
    model.eval()
    metrics_calc = MetricsCalculator()

    print("\nEvaluating on test set...")
    print("=" * 50)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Update metrics
            metrics_calc.update(outputs, labels)

    # Compute final metrics
    metrics = metrics_calc.compute()

    return metrics


def save_predictions(model, test_loader, device, output_path):
    """
    Save model predictions for further analysis.

    Args:
        model: The trained model
        test_loader: DataLoader for test set
        device: torch device
        output_path: Path to save predictions
    """
    model.eval()

    all_outputs = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Convert outputs to predictions (positive = spoof (1), negative = live (0))
            predictions = (outputs > 0).long()

            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Save to file
    with open(output_path, "w") as f:
        f.write("output,label,prediction\n")
        for output, label, prediction in zip(all_outputs, all_labels, all_predictions):
            f.write(f"{output:.4f},{label},{prediction}\n")

    print(f"Predictions saved to {output_path}")


def main(config):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get transforms
    transforms_dict = get_transforms(config["transform_type"])
    if transforms_dict is None:
        raise ValueError(f"Transform type {config['transform_type']} not supported")

    # Get data loaders (only need test loader)
    _, _, test_loader, label_map = get_data_loaders(
        train_sensor_path=config["train_sensor"],
        test_sensor_path=config["test_sensor"],
        transform=transforms_dict,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        val_split=config["val_split"],
        seed=config["seed"],
    )

    # Get model checkpoint path
    checkpoint_path = config.get("eval_checkpoint_path")
    if checkpoint_path is None:
        # Use best_model_name if eval_checkpoint_path not specified
        checkpoint_path = Path(config["checkpoint_dir"]) / config["best_model_name"]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"\nLoading model from {checkpoint_path}...")

    # Initialize model
    model = MosFPAD().to(device)
    model = load_checkpoint(model, checkpoint_path)

    # Print model info
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Evaluate
    test_metrics = evaluate(model, test_loader, device)

    # Print metrics
    print_metrics(test_metrics, phase="Test")

    # Save test results
    results_file = results_dir / config["test_results_name"]
    save_metrics(test_metrics, results_file, phase="Test")

    # Optionally save predictions
    if config.get("save_predictions", False):
        predictions_file = results_dir / "test_predictions.txt"
        save_predictions(model, test_loader, device, predictions_file)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MoSFPAD model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed for reproducibility
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])

    main(config)
