import torch
from fmod.base.util.logging import lgm

"""
Contains complex contractions wrapped into jit for harmonic layers
"""

@torch.jit.script
def contract_diagonal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    res = torch.einsum("bixy,kixy->bkxy", ac, bc)
    lgm().log( f" ++++++++ contract diagonal: bixy{tuple(a.shape)}, kixy{tuple(b.shape)} -> bkxy{tuple(res.shape)}" )
    return torch.view_as_real(res)

@torch.jit.script
def contract_dhconv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    res = torch.einsum("bixy,kix->bkxy", ac, bc)
    lgm().log(f" ++++++++ contract dhconv: bixy{tuple(a.shape)}, kix{tuple(b.shape)} -> bkxy{tuple(res.shape)}")
    return torch.view_as_real(res)

@torch.jit.script
def contract_blockdiag(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    res = torch.einsum("bixy,kixyz->bkxz", ac, bc)
    lgm().log(f" ++++++++ contract blockdiag: bixy{tuple(a.shape)}, kixyz{tuple(b.shape)} -> bkxy{tuple(res.shape)}")
    return torch.view_as_real(res)

# Helper routines for the non-linear FNOs (Attention-like)
@torch.jit.script
def compl_mul1d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixs,ior->srbox", a, b)
    res = torch.stack([tmp[0,0,...] - tmp[1,1,...], tmp[1,0,...] + tmp[0,1,...]], dim=-1) 
    return res

@torch.jit.script
def compl_mul1d_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bix,io->box", ac, bc)
    res = torch.view_as_real(resc)
    return res

@torch.jit.script
def compl_muladd1d_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    res = compl_mul1d_fwd(a, b) + c
    return res

@torch.jit.script
def compl_muladd1d_fwd_c(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tmpcc = torch.view_as_complex(compl_mul1d_fwd_c(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)

# Helper routines for FFT MLPs

@torch.jit.script
def compl_mul2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixys,ior->srboxy", a, b)
    res = torch.stack([tmp[0,0,...] - tmp[1,1,...], tmp[1,0,...] + tmp[0,1,...]], dim=-1) 
    return res


@torch.jit.script
def compl_mul2d_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,io->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res

@torch.jit.script
def compl_muladd2d_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    res = compl_mul2d_fwd(a, b) + c
    return res

@torch.jit.script
def compl_muladd2d_fwd_c(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tmpcc = torch.view_as_complex(compl_mul2d_fwd_c(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)

@torch.jit.script
def real_mul2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.einsum("bixy,io->boxy", a, b)
    return out

@torch.jit.script
def real_muladd2d_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return compl_mul2d_fwd_c(a, b) + c

