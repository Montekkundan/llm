---
title: picoLLM v1 Demo
emoji: "🧠"
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
models:
  - montekkundan/picollm-v1
---

# picoLLM v1 Demo

This Space serves `montekkundan/picollm-v1` through the picoLLM web UI.

- Hardware target: free `cpu-basic`
- Model mount path: `/data/picollm`
- Runtime mode: `python -m picollm.accelerated.chat.web -i sft --device-type cpu`

This is meant for lightweight testing on the Hugging Face website itself. It will sleep when idle on free hardware, and CPU inference will be slower than a GPU deployment.
