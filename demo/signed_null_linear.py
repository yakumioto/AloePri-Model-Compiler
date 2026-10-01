import time
import torch
import torch.nn.functional as F


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signed_null_support(p, q):
    require(p.ndim == q.ndim == 2 and p.dtype == q.dtype == torch.float64 and p.shape[0] == q.shape[1] and p.shape[1] == q.shape[0], "signed-null P/Q geometry/dtype mismatch")
    d, big_d = p.shape
    require(d > 0 and big_d > d and (big_d - d) % 4 == 0 and torch.isfinite(p).all().item() and torch.isfinite(q).all().item(), "signed-null dimensions/finite invalid")
    diagonal = torch.diag(p[:, :d])
    require(torch.all((diagonal == 1) | (diagonal == -1)).item() and torch.any(diagonal == -1).item(), "signed-null core is not nontrivial signed diagonal")
    require(torch.equal(p[:, :d], torch.diag(diagonal)) and torch.equal(q[:d], torch.diag(diagonal)), "signed-null core has mixing or inconsistent signs")
    active_p = torch.any(p[:, d:] != 0, dim=0)
    active_q = torch.any(q[d:] != 0, dim=1)
    require(torch.all(active_p ^ active_q).item() and active_p.sum().item() == active_q.sum().item() == (big_d - d) // 2, "signed-null auxiliary support is not complementary/nonzero")
    return active_p, active_q


class SignedColumnBlockedLinear(torch.nn.Linear):
    def __init__(self, weight, d, active_p):
        super().__init__(weight.shape[1], weight.shape[0], bias=False, device="meta", dtype=torch.float32)
        require(weight.dtype == torch.float32 and weight.device.type == "cpu" and weight.shape[1] > d, "signed-null input weight contract invalid")
        self.weight = weight if isinstance(weight, torch.nn.Parameter) else torch.nn.Parameter(weight, requires_grad=False)
        self.d = d
        self.register_buffer("_active_p", active_p, persistent=False)
        self.core_calls = self.aux_calls = 0
        self.peak_f32_temporary_weight_bytes = 0
        self.elapsed_seconds = 0.0
        require(torch.count_nonzero(weight[:, d:][:, active_p]).item() == 0, "signed-null input active-aux weight is nonzero")

    @torch.inference_mode()
    def forward(self, encoded):
        started = time.perf_counter()
        require(encoded.dtype == torch.float32 and encoded.shape[-1] == self.in_features and torch.isfinite(encoded).all().item(), "signed-null input is not full-D F32/finite")
        core_weight = self.weight[:, :self.d].contiguous()
        core = F.linear(encoded[..., :self.d].contiguous(), core_weight, None)
        aux = F.linear(encoded[..., self.d:], self.weight[:, self.d:], None)
        self.core_calls += 1
        self.aux_calls += 1
        require(torch.count_nonzero(encoded[..., self.d:][..., ~self._active_p]).item() == 0, "signed-null encoded inactive auxiliary state is nonzero")
        require(torch.isfinite(core).all().item() and torch.isfinite(aux).all().item() and torch.count_nonzero(aux).item() == 0, "signed-null full auxiliary GEMM product is not zero")
        self.peak_f32_temporary_weight_bytes = max(self.peak_f32_temporary_weight_bytes, core_weight.numel()*4)
        result = core + aux
        require(result.dtype == torch.float32 and torch.isfinite(result).all().item(), "signed-null input Linear output invalid")
        self.elapsed_seconds += time.perf_counter() - started
        return result


class SignedRowBlockedLinear(torch.nn.Linear):
    def __init__(self, weight, d, active_p):
        super().__init__(weight.shape[1], weight.shape[0], bias=False, device="meta", dtype=torch.float32)
        require(weight.dtype == torch.float32 and weight.device.type == "cpu" and weight.shape[0] > d, "signed-null output weight contract invalid")
        self.weight = weight if isinstance(weight, torch.nn.Parameter) else torch.nn.Parameter(weight, requires_grad=False)
        self.d = d
        self.register_buffer("_active_p", active_p, persistent=False)
        self.core_calls = self.aux_calls = 0
        self.peak_f32_temporary_weight_bytes = 0
        self.elapsed_seconds = 0.0
        require(torch.count_nonzero(weight[d:][~active_p]).item() == 0, "signed-null inactive output auxiliary weight is nonzero")

    @torch.inference_mode()
    def forward(self, projected):
        started = time.perf_counter()
        require(projected.dtype == torch.float32 and projected.shape[-1] == self.in_features and torch.isfinite(projected).all().item(), "signed-null output Linear input invalid")
        core = F.linear(projected, self.weight[:self.d], None)
        aux = F.linear(projected, self.weight[self.d:], None)
        self.core_calls += 1
        self.aux_calls += 1
        require(torch.isfinite(core).all().item() and torch.isfinite(aux).all().item() and torch.count_nonzero(aux[..., ~self._active_p]).item() == 0, "signed-null output auxiliary support product invalid")
        result = torch.cat([core, aux], dim=-1)
        require(result.dtype == torch.float32 and result.shape[-1] == self.out_features, "signed-null output is not full-D F32")
        self.elapsed_seconds += time.perf_counter() - started
        return result
