import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loaders import get_data_loaders
from metrics import MetricsCalculator, print_metrics, save_metrics
from model import MosFPAD
from transforms import get_transforms


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    metrics_calc = MetricsCalculator()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Convert labels from {0, 1} to {-1, +1} for MarginRankingLoss
        targets = 2 * labels - 1
        # MarginRankingLoss requires (input1, input2, target)
        # Use zeros as input2 (decision boundary at 0)
        zeros = torch.zeros_like(outputs)
        loss = criterion(outputs, zeros, targets)

        loss.backward()
        utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        # Update metrics
        metrics_calc.update(outputs.detach(), labels.long())

        # Get current metrics
        current_metrics = metrics_calc.compute()

        pbar.set_postfix(
            {
                "loss": f"{running_loss / metrics_calc.total:.4f}",
                "acc": f"{current_metrics['accuracy']:.2f}%",
            }
        )

    epoch_loss = running_loss / metrics_calc.total
    metrics = metrics_calc.compute()

    return epoch_loss, metrics


def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    metrics_calc = MetricsCalculator()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)

            # Convert labels from {0, 1} to {-1, +1} for MarginRankingLoss
            targets = 2 * labels - 1
            # MarginRankingLoss requires (input1, input2, target)
            # Use zeros as input2 (decision boundary at 0)
            zeros = torch.zeros_like(outputs)
            loss = criterion(outputs, zeros, targets)

            running_loss += loss.item() * labels.size(0)

            # Update metrics
            metrics_calc.update(outputs, labels.long())

            # Get current metrics
            current_metrics = metrics_calc.compute()

            pbar.set_postfix(
                {
                    "loss": f"{running_loss / metrics_calc.total:.4f}",
                    "acc": f"{current_metrics['accuracy']:.2f}%",
                }
            )

    epoch_loss = running_loss / metrics_calc.total
    metrics = metrics_calc.compute()

    return epoch_loss, metrics


def test(model, test_loader, device):
    model.eval()
    metrics_calc = MetricsCalculator()

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Update metrics
            metrics_calc.update(outputs, labels)

    # Compute final metrics
    metrics = metrics_calc.compute()

    # Print metrics
    print_metrics(metrics, phase="Test")

    return metrics


def save_checkpoint(model, optimizer, epoch, best_acc, save_path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def get_scheduler(optimizer, config):
    """Create learning rate scheduler based on config"""
    if not config.get("use_scheduler", False):
        return None

    scheduler_type = config.get("scheduler_type", "step")

    if scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("lr_step_size", 50),
            gamma=config.get("lr_gamma", 0.1),
        )
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config.get("lr_min", 1e-6)
        )
    elif scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.get("lr_gamma", 0.1),
            patience=config.get("lr_patience", 10),
            verbose=True,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def main(config):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    config_dir = Path(config["config_dir"])
    log_dir = Path(config["log_dir"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    results_dir = Path(config["results_dir"])

    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config to config directory
    with open(config_dir / config["config_name"], "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # Get transforms
    transforms_dict = get_transforms(config["transform_type"])
    if transforms_dict is None:
        raise ValueError(f"Transform type {config['transform_type']} not supported")

    # Get data loaders
    train_loader, val_loader, test_loader, label_map = get_data_loaders(
        train_sensor_path=config["train_sensor"],
        test_sensor_path=config["test_sensor"],
        transform=transforms_dict,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        val_split=config["val_split"],
        seed=config["seed"],
    )

    # Initialize model
    model = MosFPAD().to(device)
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Loss and optimizer - use MarginRankingLoss
    criterion = nn.MarginRankingLoss(margin=config.get("hinge_margin", 1.0))

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Learning rate scheduler (optional)
    scheduler = get_scheduler(optimizer, config)
    if scheduler is not None:
        print(f"Using {config.get('scheduler_type', 'step')} learning rate scheduler")
    else:
        print("No learning rate scheduler")

    # Training loop
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'=' * 50}")

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)

        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["accuracy"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("BPCER/train", train_metrics["bpcer"], epoch)
        writer.add_scalar("BPCER/val", val_metrics["bpcer"], epoch)
        writer.add_scalar("APCER/train", train_metrics["apcer"], epoch)
        writer.add_scalar("APCER/val", val_metrics["apcer"], epoch)
        writer.add_scalar("ACE/train", train_metrics["ace"], epoch)
        writer.add_scalar("ACE/val", val_metrics["ace"], epoch)
        writer.add_scalar("Learning_rate", current_lr, epoch)

        print(
            f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%, "
            f"Train ACE: {train_metrics['ace']:.2f}%"
        )
        print(
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, "
            f"Val ACE: {val_metrics['ace']:.2f}%"
        )
        print(f"Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_path = checkpoint_dir / config["best_model_name"]
            save_checkpoint(model, optimizer, epoch, best_val_acc, save_path)
            print(f"New best validation accuracy: {best_val_acc:.2f}%")

        # Save checkpoint every N epochs
        if epoch % config["save_freq"] == 0:
            save_path = checkpoint_dir / f"{config['checkpoint_prefix']}{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, best_val_acc, save_path)

    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"Training completed in {elapsed_time / 3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'=' * 50}")

    # Test on best model
    print("\nLoading best model for testing...")
    checkpoint = torch.load(checkpoint_dir / config["best_model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = test(model, test_loader, device)

    # Save test results
    results_file = results_dir / config["test_results_name"]
    save_metrics(test_metrics, results_file, phase="Test")

    print(f"\nAll results saved to {results_dir}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MoSFPAD model")
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
