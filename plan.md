1. **Add `entropic_smoothing_alpha` to MDLMRequest & model arguments**
   - Update `MDLMRequest` in `src/models/diffusion/modeling_diffusion.py` to include `entropic_smoothing_alpha: float = 0.0`.
   - Update `generate` signature in `DiffusionModelForConditionalGeneration` (around line 741) to accept `entropic_smoothing_alpha: float = 0.0`.

2. **Implement Token-Level Entropic Smoothing (TLES) in standard generation**
   - In `generate` (around line 2542-2660), inside the temperature modification block, compute entropy of the distribution, and update `current_temperature` per token: `current_temperature = current_temperature + entropic_smoothing_alpha / (entropy + 1e-5)` (or effectively modify logits directly).
   - Actually, since `temperature` modifies logits before sampling (by dividing them), we can compute a per-token scaling factor `eff_temperature` directly using `1.0 + entropic_smoothing_alpha / (entropy + 1e-5)` and apply it to `logits`.
   - Wait, `entropy` is computed per-token? No, `entropy` of `logits` at step `t` is scalar per token position, so `logits` is `(batch, seq_len, vocab_size)`, entropy is `(batch, seq_len, 1)`. The temperature modifications already handle tensors.
   - So we will add:
     ```python
     if entropic_smoothing_alpha > 0.0:
         probs = F.softmax(logits, dim=-1)
         entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1, keepdim=True)
         tles_temp = 1.0 + entropic_smoothing_alpha / (entropy + 1e-5)
         current_temperature = current_temperature * tles_temp
     ```

3. **Implement TLES in dynamic batching generation**
   - Apply the same logic in `MDLMContinuousBatchingManager.step()` (around line 4543), using `req.entropic_smoothing_alpha`.
     ```python
     if req.entropic_smoothing_alpha > 0.0:
         probs = F.softmax(req_logits, dim=-1)
         entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1, keepdim=True)
         tles_temp = 1.0 + req.entropic_smoothing_alpha / (entropy + 1e-5)
         current_temperature = current_temperature * tles_temp
     ```
   - Make sure `if req.temperature > 0 or ... or req.entropic_smoothing_alpha > 0` condition is updated.

4. **Write tests**
   - Add tests for `entropic_smoothing_alpha` in standard `generate`.
   - Add tests for `entropic_smoothing_alpha` in `generate_dynamic_batch`.

5. **Run Pre-Commit & Tests**
   - Ensure the tests pass and coverage is maintained (target 90%).
   - Check pre-commit steps.

6. **Submit**
