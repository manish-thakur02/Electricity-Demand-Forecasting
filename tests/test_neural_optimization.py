import pytest
import torch
from pytorch_forecasting.models.nn.rnn import LSTM
from forecasting.neural import uniform_lstm


@pytest.mark.parametrize("lengths", [[8, 8, 8], [8, 5, 2], [8, 0, 2]])
def test_dense_and_packed_outputs_states_and_gradients_agree(lengths):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    original = LSTM(4, 8, num_layers=2, batch_first=True, dropout=0)
    rng = torch.random.get_rng_state().clone()
    optimized = uniform_lstm(original)
    assert torch.equal(rng, torch.random.get_rng_state())
    values = torch.randn(3, 10, 4)
    lengths = torch.tensor(lengths)
    x1, x2 = values.clone().requires_grad_(), values.clone().requires_grad_()
    y1, (h1, c1) = original(x1, lengths=lengths, enforce_sorted=False)
    y2, (h2, c2) = optimized(x2, lengths=lengths, enforce_sorted=False)
    for a, b in [(y1, y2), (h1, h2), (c1, c2)]:
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
    (y1.square().mean() + h1.square().mean()).backward()
    (y2.square().mean() + h2.square().mean()).backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-4, atol=1e-6)
    for a, b in zip(original.parameters(), optimized.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)
