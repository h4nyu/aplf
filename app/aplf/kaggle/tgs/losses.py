import torch.nn.functional as F


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    label = F.softmax(target_logits, dim=1).argmax(dim=1)
    return F.cross_entropy(input_logits, label, size_average=True)
