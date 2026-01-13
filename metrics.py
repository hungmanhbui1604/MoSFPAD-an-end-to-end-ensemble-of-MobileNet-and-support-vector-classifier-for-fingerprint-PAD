import torch


class MetricsCalculator:
    """Calculate various metrics for fingerprint PAD"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.correct = 0
        self.total = 0
        self.live_correct = 0
        self.live_total = 0
        self.spoof_correct = 0
        self.spoof_total = 0
    
    def update(self, outputs, labels):
        """
        Update metrics with batch results
        
        Args:
            outputs: model outputs (raw scores)
            labels: ground truth labels (0 for live, 1 for spoof)
        """
        # Convert outputs to predictions (positive = spoof (1), negative = live (0))
        predictions = (outputs > 0).long()
        
        # Update overall accuracy
        self.correct += (predictions == labels).sum().item()
        self.total += labels.size(0)
        
        # Update live and spoof metrics
        live_mask = (labels == 0)
        spoof_mask = (labels == 1)
        
        if live_mask.sum() > 0:
            self.live_correct += (predictions[live_mask] == labels[live_mask]).sum().item()
            self.live_total += live_mask.sum().item()
        
        if spoof_mask.sum() > 0:
            self.spoof_correct += (predictions[spoof_mask] == labels[spoof_mask]).sum().item()
            self.spoof_total += spoof_mask.sum().item()
    
    def compute(self):
        """
        Compute all metrics
        
        Returns:
            dict: Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = 100 * self.correct / self.total if self.total > 0 else 0
        
        # BPCER: Bonafide Presentation Classification Error Rate
        # Percentage of misclassified live samples
        metrics['bpcer'] = 100 * (1 - self.live_correct / self.live_total) if self.live_total > 0 else 0
        
        # APCER: Attack Presentation Classification Error Rate
        # Percentage of misclassified spoof samples
        metrics['apcer'] = 100 * (1 - self.spoof_correct / self.spoof_total) if self.spoof_total > 0 else 0
        
        # ACE: Average Classification Error
        metrics['ace'] = (metrics['apcer'] + metrics['bpcer']) / 2

        return metrics


def print_metrics(metrics, phase='Test'):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        phase: Phase name (e.g., 'Train', 'Val', 'Test')
    """
    print(f"\n{phase} Metrics:")
    print(f"{'='*50}")
    print(f"Accuracy:    {metrics['accuracy']:.2f}%")
    print(f"BPCER:       {metrics['bpcer']:.2f}%")
    print(f"APCER:       {metrics['apcer']:.2f}%")
    print(f"ACE:         {metrics['ace']:.2f}%")
    print(f"{'='*50}")


def save_metrics(metrics, filepath, phase='Test'):
    """
    Save metrics to file
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save file
        phase: Phase name
    """
    with open(filepath, 'w') as f:
        f.write(f"{phase} Metrics\n")
        f.write(f"{'='*50}\n")
        f.write(f"Accuracy:    {metrics['accuracy']:.2f}%\n")
        f.write(f"BPCER:       {metrics['bpcer']:.2f}%\n")
        f.write(f"APCER:       {metrics['apcer']:.2f}%\n")
        f.write(f"ACE:         {metrics['ace']:.2f}%\n")
        f.write(f"{'='*50}\n")