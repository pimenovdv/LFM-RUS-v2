1. **Add `classifier_fn` and `classifier_scale` to `generate` parameters:**
   - Modify `src/models/diffusion/modeling_diffusion.py` to include `classifier_fn: Optional[callable] = None` and `classifier_scale: float = 1.0` and `classifier_schedule: str = "constant"` and `min_classifier_scale: float = 0.0`.
   - Update type hint imports for Callable.

2. **Implement Classifier-Guided Sampling logic in the generation loop:**
   - In `generate`, during the logit computation, check if `classifier_fn` is provided and `classifier_scale > 0.0`.
   - If so, ensure `logits` require gradients (`logits.requires_grad_(True)`). However, since we are under `@torch.no_grad()`, we need to handle this carefully.
   - We might need to locally enable gradients for the `logits` to compute `classifier_fn` gradient with respect to `logits`.
   - Compute `loss = classifier_fn(logits, x)` or something similar.
   - Compute `grad = torch.autograd.grad(loss, logits)[0]`.
   - Modify the logits using the gradient: `guided_logits = logits + classifier_scale * grad`.
   - Add schedule logic for `classifier_scale`.

3. **Write tests for Classifier-Guided Sampling:**
   - Create a test case in `tests/` to verify that `generate` uses the `classifier_fn` correctly and changes the output logits.

4. **Pre-commit checks**
   - Run coverage and tests to ensure no regressions and 90% coverage threshold is met.

5. **Update `todo.md`:**
   - Mark Step 56 as complete.
