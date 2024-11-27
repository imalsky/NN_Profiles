# losses.py

import torch
import torch.nn as nn

class FluxConservationLoss(nn.Module):
    def __init__(self, lambda_conservation=0.1):
        super(FluxConservationLoss, self).__init__()
        self.lambda_conservation = lambda_conservation
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets, boundary_fluxes):
        outputs = outputs.squeeze(-1)
        targets = targets.squeeze(-1)
        boundary_fluxes = (boundary_fluxes[0].squeeze(-1), boundary_fluxes[1].squeeze(-1))
        
        # Compute standard MSE loss
        mse = self.mse_loss(outputs, targets)

        # Compute conservation loss
        predicted_flux_difference = outputs[:, -1] - outputs[:, 0]  # Shape: [batch_size]
        true_flux_difference = boundary_fluxes[1] - boundary_fluxes[0]  # Shape: [batch_size]
        conservation_loss = torch.abs(predicted_flux_difference - true_flux_difference)

        # Total loss
        total_loss = mse + self.lambda_conservation * conservation_loss.mean()
        return total_loss
