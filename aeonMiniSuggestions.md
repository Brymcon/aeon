
For AeonMini on an RTX 3090, here are some additional suggestions:

1. **Memory-to-compute ratio optimization** - The 3090's 24GB VRAM allows for larger batch sizes, but you'll get better throughput by using slightly smaller batches (4096-8192) with higher gradient accumulation steps.

2. **Temperature management** - For extended training sessions, consider setting a custom fan curve as the 3090 can throttle during multi-hour runs.

3. **CPU-GPU bottlenecks** - Ensure your environment simulation isn't CPU-bound; the 3090 can process neural network operations much faster than most environments can generate experiences.

4. **CUDA stream optimization** - Enable environment vectorization with multiple CUDA streams to better saturate the GPU compute resources.

5. **Single vs half precision** - On complex tasks with high-dimensional observations, you might get better results with FP32, despite the speed advantage of FP16.

6. **VRAM allocation patterns** - Pre-allocate your replay buffers at the start rather than growing them dynamically to avoid memory fragmentation.

7. **Paired with AMP** - The 3090 has specialized tensor cores; when used with automatic mixed precision, you can achieve 2.5-3x throughput without stability issues.

8. **Efficient dataloader workers** - For larger environments, set num_workers=4 in your DataLoader configuration to keep the GPU fed with experiences.

9. **Multi-environment scaling** - The 3090 can effectively handle 64-128 parallel environments for data collection before hitting diminishing returns.

10. **Monitoring utilization** - Use `nvidia-smi` with the `-l` flag during training to ensure GPU utilization stays above 80% for optimal efficiency.
