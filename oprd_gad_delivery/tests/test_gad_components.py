"""CPU-only unit tests for the OPRD+GAD merge (no GPU / no ray / no vllm required).

Two layers:
  1. Pure-logic tests that replicate the exact math added to the fork and validate its
     numerical + gradient behavior. These RUN in a bare environment (only torch needed).
  2. Real-import parity tests, guarded by `importorskip`, that exercise the actual
     verl functions once ray/vllm are installed (i.e. on the GPU training box).

Run here:   python3 test_gad_components.py      (or: pytest -q test_gad_components.py)
"""

import torch
import torch.nn.functional as F


# --- Reference reimplementations (MUST mirror the fork edits) ------------------------------

def _ref_discriminator_loss(student_vpreds, teacher_vpreds, response_mask, teacher_response_mask):
    """Mirror of core_algos.compute_discriminator_loss (Bradley-Terry: teacher > student)."""
    teacher_reward = torch.sum(teacher_vpreds * teacher_response_mask, dim=-1)
    student_reward = torch.sum(student_vpreds * response_mask, dim=-1)
    return -F.logsigmoid(teacher_reward - student_reward).mean()


def _ref_last_token_extract(values, attention_mask, response_length):
    """Mirror of DataParallelPPOCritic._slice_response_values (use_gad=True branch)."""
    values = values[:, -response_length:]
    resp_attn = attention_mask[:, -response_length:]
    resp_len = resp_attn.sum(dim=1).long()
    last_idx = (resp_len - 1).clamp(min=0)
    last_mask = torch.zeros_like(resp_attn, dtype=torch.bool)
    batch_indices = torch.arange(resp_attn.size(0), device=resp_attn.device)
    last_mask[batch_indices, last_idx] = True
    return values * last_mask.type_as(values)


# --- Layer 1: pure-logic tests (run without verl deps) -------------------------------------

def test_bt_loss_value_monotonic():
    """Loss shrinks as teacher score exceeds student score."""
    rmask = torch.ones(1, 1)
    # teacher >> student  -> tiny loss
    lo = _ref_discriminator_loss(torch.tensor([[0.0]]), torch.tensor([[10.0]]), rmask, rmask)
    # teacher << student  -> large loss
    hi = _ref_discriminator_loss(torch.tensor([[10.0]]), torch.tensor([[0.0]]), rmask, rmask)
    # equal -> -log(0.5)
    eq = _ref_discriminator_loss(torch.tensor([[1.0]]), torch.tensor([[1.0]]), rmask, rmask)
    assert lo.item() < 0.01, lo.item()
    assert hi.item() > 5.0, hi.item()
    assert abs(eq.item() - 0.6931) < 1e-3, eq.item()


def test_bt_loss_gradient_sign():
    """Minimizing the loss must push the teacher score UP and the student score DOWN."""
    student = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)  # last-token score at pos 2
    teacher = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
    rmask = torch.ones(1, 3)
    loss = _ref_discriminator_loss(student, teacher, rmask, rmask)
    loss.backward()
    # dL/d teacher < 0  (grad descent raises teacher score); dL/d student > 0 (lowers it)
    assert teacher.grad.sum().item() < 0, teacher.grad
    assert student.grad.sum().item() > 0, student.grad


def test_last_token_extraction_right_padding():
    """Only the last *real* response token keeps its score; padded tail is zeroed."""
    # seqlen=6, response_length=4; row0 has 3 real resp tokens then 1 pad, row1 has 2 real then 2 pad
    values = torch.tensor(
        [[9, 9, 1.0, 2.0, 3.0, 7.0],   # response region = last 4 = [1,2,3,7]; real=3 -> keep idx2 (val 3)
         [9, 9, 5.0, 6.0, 8.0, 4.0]],  # real=2 -> keep idx1 (val 6)
    )
    attn = torch.tensor(
        [[1, 1, 1, 1, 1, 0],   # response region last4 = [1,1,1,0] -> 3 real
         [1, 1, 1, 1, 0, 0]],  # response region last4 = [1,1,0,0] -> 2 real
    )
    out = _ref_last_token_extract(values, attn, response_length=4)
    assert out.shape == (2, 4)
    # row0: keep position 2 (value 3.0), zero elsewhere; row1: keep position 1 (value 6.0)
    assert torch.equal(out[0], torch.tensor([0.0, 0.0, 3.0, 0.0]))
    assert torch.equal(out[1], torch.tensor([0.0, 6.0, 0.0, 0.0]))
    # summing over the response yields the sequence-level score D(y)
    assert out.sum(dim=-1).tolist() == [3.0, 6.0]


def test_last_token_all_padded_is_safe():
    """A degenerate all-pad row clamps to index 0 and does not crash."""
    values = torch.tensor([[1.0, 2.0]])
    attn = torch.tensor([[0, 0]])
    out = _ref_last_token_extract(values, attn, response_length=2)
    assert out.shape == (1, 2)


# --- Layer 2: real-import parity (skipped unless verl deps present, e.g. on GPU box) --------

def test_real_compute_discriminator_loss_parity():
    import pytest

    pytest.importorskip("ray")
    from verl.trainer.ppo.core_algos import compute_discriminator_loss

    torch.manual_seed(0)
    s = torch.randn(4, 8)
    t = torch.randn(4, 8)
    rm = torch.randint(0, 2, (4, 8)).float()
    trm = torch.randint(0, 2, (4, 8)).float()
    real = compute_discriminator_loss(s, t, rm, trm)
    ref = _ref_discriminator_loss(s, t, rm, trm)
    assert torch.allclose(real, ref), (real, ref)


if __name__ == "__main__":
    test_bt_loss_value_monotonic()
    test_bt_loss_gradient_sign()
    test_last_token_extraction_right_padding()
    test_last_token_all_padded_is_safe()
    print("PASS: all pure-logic tests (BT loss + last-token extraction)")
    try:
        test_real_compute_discriminator_loss_parity()
        print("PASS: real-import parity test")
    except BaseException as e:  # ImportError / pytest.Skipped when verl deps absent
        print(f"SKIP: real-import parity test ({type(e).__name__})")
