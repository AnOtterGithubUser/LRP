import pytest
from pylrp import layers
import torch


def test_LRPMaxPool2d_forward_pass_should_record_the_indices_of_the_maximum_elements_in_input():
    # Given
    expected_indices = torch.tensor([[[[0, 6], [12, 14]]]])
    x = torch.zeros(16)
    x[expected_indices] = 1
    x = x.view(1, 1, 4, 4)
    max_pool_layer = layers.LRPMaxPool2d(kernel_size=2)

    # When
    max_pool_layer(x)
    actual_indices = max_pool_layer.indices

    # Then
    assert(torch.all(torch.eq(expected_indices, actual_indices)))


def test_LRPMaxPool2d_backward_relevance_should_distribute_relevance_on_maximum_elements_before_pooling():
    # Given
    R = torch.zeros((2, 2))
    R[:] = 3
    R = R.view(1, 1, 2, 2)
    indices = torch.tensor([[[[0, 6], [12, 14]]]])
    x = torch.zeros(16)
    x[indices] = 1
    x = x.view(1, 1, 4, 4)
    max_pool_layer = layers.LRPMaxPool2d(kernel_size=2)
    expected_relevance = torch.zeros(16)
    expected_relevance[indices] = 3
    expected_relevance = expected_relevance.view(1, 1, 4, 4)

    # When
    max_pool_layer(x)
    actual_relevance = max_pool_layer.backward_relevance(R)

    # Then
    assert(torch.all(torch.eq(expected_relevance, actual_relevance)))
