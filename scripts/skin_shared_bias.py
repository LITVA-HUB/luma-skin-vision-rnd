"""Paired mean-bias objectives with a fixed single-image inference model."""
import torch

ARMS = ['individual', 'consistency', 'shared_half', 'shared_only']


def paired_objective(prediction, reference, arm):
    if len(prediction) % 2:
        raise ValueError('Expected even paired batch')
    n = len(prediction)//2
    if prediction.shape != reference.shape or not torch.equal(reference[:n], reference[n:]):
        raise ValueError('Paired reference mismatch')
    individual = (prediction-reference).square().mean()
    if arm == 'individual':
        return individual
    a, b = prediction[:n], prediction[n:]
    shared = ((a+b)/2-reference[:n]).square().mean()
    if arm == 'consistency':
        return individual + (a-b).square().mean()/4
    if arm == 'shared_half':
        return .5*individual+.5*shared
    if arm == 'shared_only':
        return shared
    raise ValueError('Unknown paired objective')
