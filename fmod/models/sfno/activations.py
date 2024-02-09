import torch
import torch.nn as nn

# complex activation functions

class ComplexCardioid(nn.Module):
    """
    Complex Cardioid activation function
    """
    def __init__(self):
        super(ComplexCardioid, self).__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = 0.5 * (1. + torch.cos(z.angle())) * z
        return out

class ComplexReLU(nn.Module):
    """
    Complex-valued variants of the ReLU activation function
    """
    def __init__(self, negative_slope=0., mode="real", bias_shape=None, scale=1.):
        super(ComplexReLU, self).__init__()
        
        # store parameters
        self.mode = mode
        if self.mode in ["modulus", "halfplane"]:
            if bias_shape is not None:
                self.bias = nn.Parameter(scale * torch.ones(bias_shape, dtype=torch.float32))
            else:
                self.bias = nn.Parameter(scale * torch.ones((1), dtype=torch.float32))
        else:
            self.bias = 0

        self.negative_slope = negative_slope
        self.act = nn.LeakyReLU(negative_slope = negative_slope)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)

        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = torch.where(zabs + self.bias > 0, (zabs + self.bias) * z / zabs, 0.0)

        elif self.mode == "cardioid":
            out = 0.5 * (1. + torch.cos(z.angle())) * z

        # elif self.mode == "halfplane":
        #     # bias is an angle parameter in this case
        #     modified_angle = torch.angle(z) - self.bias
        #     condition = torch.logical_and( (0. <= modified_angle), (modified_angle < torch.pi/2.) )
        #     out = torch.where(condition, z, self.negative_slope * z)

        elif self.mode == "real":
            zr = torch.view_as_real(z)
            outr = zr.clone()
            outr[..., 0] = self.act(zr[..., 0])
            out = torch.view_as_complex(outr)
        else:
            raise NotImplementedError
            
        return out