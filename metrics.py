import torch

def accuracy(distances, y, step=0.01):
    min_threshold_d = min(distances)
    max_threshold_d = max(distances)
    max_acc = 0
    best_threshold = min_threshold_d
    same_id = (y == 1)

    for threshold_d in torch.arange(min_threshold_d, max_threshold_d+step, step):
        true_positive = (distances <= threshold_d) & (same_id)
        true_positive_rate = true_positive.sum().float() / same_id.sum().float()
        true_negative = (distances > threshold_d) & (~same_id)
        true_negative_rate = true_negative.sum().float() / (~same_id).sum().float()

        acc = 0.5 * (true_negative_rate + true_positive_rate)
        if acc > max_acc:
            max_acc = acc
            best_threshold = threshold_d
    return max_acc, best_threshold
