# picoLLM Experiment Report

Use this template after running checkpoint comparison, latency measurement, and any optional benchmark or safety checks.

## 1. Run identity

- Date:
- Student / author:
- Code revision:
- Hardware:
- Model path:
- Tokenizer path:
- Training recipe:

## 2. Research question

What was this run trying to improve or test?

## 3. Configuration summary

- Dataset(s):
- Batch size:
- Gradient accumulation:
- Learning rate:
- Warmup:
- Max steps:
- Sequence length:
- Telemetry:

## 4. Training summary

- Final train loss:
- Validation loss or eval loss:
- Runtime:
- Approximate cost:
- Notes on throughput / step time:

## 5. Checkpoint comparison

Reference the JSON from `picollm.eval.compare_checkpoints`.

- Base checkpoint:
- Compared checkpoint:
- Most obvious improvements:
- Most obvious regressions:

## 6. Latency summary

Reference the JSON from `picollm.eval.latency_benchmark`.

- Mean latency:
- Median latency:
- Mean tokens per second:
- Notes:

## 7. Safety / red-team summary

Reference the JSON from `picollm.eval.safety_red_team`.

- Harmful prompts handled well:
- Concerning failures:
- Over-refusals:
- Notes:

## 8. Optional formal benchmark results

Reference the output from `picollm.eval.run_lm_eval`.

- Tasks run:
- Key scores:
- Interpretation:

## 9. Conclusion

Was the run worth its cost? What should change in the next run?
