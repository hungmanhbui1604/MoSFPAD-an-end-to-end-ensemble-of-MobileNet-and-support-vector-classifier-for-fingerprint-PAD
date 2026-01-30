import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    """
    Hinge Loss for binary classification.

    Loss = max(0, margin - y * f(x))

    where:
        - y ∈ {-1, +1} is the label
        - f(x) is the model output
        - margin is the margin parameter (default 1.0)

    This is equivalent to the SVM hinge loss.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs: Model predictions (batch_size,) - raw scores
            targets: Labels in {-1, +1} (batch_size,)

        Returns:
            loss: Scalar hinge loss
        """
        # Hinge loss: max(0, margin - y * f(x))
        loss = torch.clamp(self.margin - targets * outputs, min=0)
        return loss.mean()
