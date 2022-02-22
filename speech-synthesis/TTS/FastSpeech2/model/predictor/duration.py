from ..module import Module
from .ops import predict


class DurationPredictor(Module):
    r"""Duration Predictor module.

    Args:
        hp (HParams): Hyper-parameters.
    """

    def __init__(self, hp):
        self.hp = hp

    def call(self, x, mask):
        r"""Returns duration prediction.

        Args:
            x (nn.Variable): Input variable of shape (B, max_len, dim).
            mask (nn.Variable, optional): Mask variable of shape
                (B, max_len, 1). Defaults to None.
        Returns:
            nn.Variable: Duration prediction variable of shape (B, max_len).
                Durations are measured in log scale.
        """
        x = predict(x, mask, self.hp, self.training)
        return x
