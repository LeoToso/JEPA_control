import torch

from jepa_lds.losses import (
    action_reconstruction_loss,
    encode_window,
    multistep_prediction_loss,
    one_step_prediction_loss,
    sigreg_loss,
)
from jepa_lds.models import LinearEncoder, LinearLatentPredictor, MultistepActionDecoder


def test_prediction_losses_shapes_and_grad():
    torch.manual_seed(0)
    obs_dim, latent_dim, action_dim, H, B = 12, 6, 1, 4, 8
    encoder = LinearEncoder(obs_dim, latent_dim)
    predictor = LinearLatentPredictor(latent_dim, action_dim)
    y_window = torch.randn(B, H + 1, obs_dim)
    a_window = torch.randn(B, H, action_dim)

    z = encode_window(encoder, y_window)
    assert z.shape == (B, H + 1, latent_dim)
    loss_ms = multistep_prediction_loss(predictor, z, a_window)
    loss_1s = one_step_prediction_loss(predictor, z, a_window)
    assert loss_ms.dim() == 0
    assert loss_1s.dim() == 0
    (loss_ms + loss_1s).backward()
    assert encoder.linear.weight.grad is not None
    assert predictor.A.weight.grad is not None


def test_multistep_prediction_loss_zero_when_predictor_matches_and_encoder_frozen():
    """If encoder and predictor jointly realize a consistent linear model,
    the recursive rollout should reproduce the encoded trajectory exactly."""
    torch.manual_seed(0)
    latent_dim, action_dim, H, B = 3, 2, 5, 4
    encoder = LinearEncoder(latent_dim, latent_dim, bias=False)
    with torch.no_grad():
        encoder.linear.weight.copy_(torch.eye(latent_dim))
    predictor = LinearLatentPredictor(latent_dim, action_dim)
    a_window = torch.randn(B, H, action_dim)
    z0 = torch.randn(B, latent_dim)
    with torch.no_grad():
        z_traj = predictor.rollout(z0, a_window)  # exact rollout under this predictor
    y_window = z_traj  # encoder is identity, so y == z
    z = encode_window(encoder, y_window)
    loss = multistep_prediction_loss(predictor, z, a_window)
    assert loss.item() < 1e-10
    loss_1s = one_step_prediction_loss(predictor, z, a_window)
    assert loss_1s.item() < 1e-10


def test_sigreg_loss_lower_for_standard_normal_than_degenerate():
    torch.manual_seed(0)
    d = 5
    z_gaussian = torch.randn(2000, d)
    z_degenerate = torch.zeros(2000, d)
    torch.manual_seed(1)
    loss_gauss = sigreg_loss(z_gaussian, n_directions=32)
    torch.manual_seed(1)
    loss_degenerate = sigreg_loss(z_degenerate, n_directions=32)
    assert loss_gauss.item() < loss_degenerate.item()
    assert loss_gauss.item() < 0.05


def test_action_reconstruction_loss_zero_when_decoder_matches():
    torch.manual_seed(0)
    latent_dim, action_dim, H, B = 4, 2, 3, 6
    decoder = MultistepActionDecoder(latent_dim, action_dim, H)
    z_window = torch.randn(B, H + 1, latent_dim)
    with torch.no_grad():
        a_window = decoder(z_window)
    loss = action_reconstruction_loss(decoder, z_window, a_window)
    assert loss.item() < 1e-10

