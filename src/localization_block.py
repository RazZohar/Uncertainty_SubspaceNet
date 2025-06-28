import torch
import torch.nn as nn

class RayIntersection(nn.Module):
    """
    Least-squares intersection of M 2-D rays.
    Inputs
    ------
    positions : (M,2) tensor of (x_i, y_i)
    bearings  : (M,)  tensor of theta_i  [rad]

    Returns
    -------
    x_hat : (2,)   estimated intersection point
    gdop  : scalar geometric dilution of precision
    """
    def forward(self, positions, bearings, eps=1e-9):
        sensor_count, pos_dim = positions.shape

        # Copy -pi/2, pi/2 to [0,pi]
        bearings = torch.remainder(bearings + torch.pi / 2, torch.pi)

        # Unit direction vectors
        d = torch.stack((torch.cos(bearings), torch.sin(bearings)), dim=1)  # (M,2)

        # Perpendiculars (+90° rotation)
        d_perp = torch.stack((-d[:,1], d[:,0]), dim=1)                      # (M,2)

        # Build normal equations
        A = d_perp.T @ d_perp                                               # (2,2)
        c = (d_perp * positions).sum(dim=1)                                 # (M,)
        b = d_perp.T @ c.float()                                                    # (2,)

        # Regularised inverse for numerical stability
        x_hat = torch.linalg.solve(A + eps*torch.eye(pos_dim, device=A.device), b)   # (2,)

        # GDOP
        gdop = torch.sqrt(torch.trace(torch.linalg.inv(A + eps*torch.eye(pos_dim))))
        return x_hat, gdop
