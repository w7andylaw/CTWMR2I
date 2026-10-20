# RTX 5090 / WSL2 GPU runtime

Use an isolated Linux/WSL2 Python 3.11 environment. The project's CPU reference environment and the original TensorFlow `dreamer` environment are separate.

```bash
cd /mnt/c/Users/Administrator/Desktop/CTWMR2I
python3.11 -m venv "$HOME/.venvs/ctwm-r2i-gpu"
source "$HOME/.venvs/ctwm-r2i-gpu/bin/activate"
python -m pip install -r requirements-gpu.txt
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/gpu_check.py --require-gpu
```

On this computer the environment is `/home/administrator/.venvs/ctwm-r2i-gpu`. Its Python was created using `/home/administrator/miniconda3/envs/dreamer/bin/python`; this does not install packages into `dreamer`.

The GPU requirements intentionally do not include `requirements.txt`: that file pins the CPU reference JAX 0.4.35 and TFP 0.25.0. Do not install both files into the same environment. Use the pinned TFP nightly distribution instead of installing `tensorflow-probability` alongside it; both provide the same import namespace.

JAX 0.6.2 with CUDA 12 pip wheels retains NumPy 1.26 compatibility for the older benchmark ecosystem. JAX 0.7.2 and later require NumPy 2. The CUDA libraries come from pip; a system CUDA toolkit is unnecessary. An inherited TensorFlow `LD_LIBRARY_PATH` can override these libraries, so unset it in the launch shell. All CT-WM model, actor and critic arrays remain float32; this runtime change does not enable mixed precision.

The check explicitly runs a float32 matrix multiplication, a float32 convolution and its gradient on the GPU, evaluates a TFP distribution, and imports the complete agent. It exits nonzero when `--require-gpu` is given and no GPU is available. It disables preallocation and requests 3% of device memory for its small probe. These checks verify runtime compatibility, not benchmark learning performance.

The probe sets `jax_default_matmul_precision=highest`. Normal CT-WM training retains R2I's default JAX matmul precision while keeping model arrays float32. On NVIDIA hardware the default may use TF32 multiplication internally; this is separate from enabling float16/bfloat16 mixed precision. For an explicit highest-precision experiment, launch training with `JAX_DEFAULT_MATMUL_PRECISION=highest` and measure its speed separately. The GPU integration tests below use the unchanged training default.

The first complete training graph can compile for several minutes on Blackwell. In the bounded joint-update check XLA printed `dot_search_space.cc` warnings that the hints matched no configurations and it was expanding its autotuning search. High CPU use with low GPU utilization during this stage is compilation, not an environment-step stall. An optional runtime comparison can use `XLA_FLAGS=--xla_gpu_autotune_level=0` to skip that search, but should measure steady-state throughput separately; this flag is not enabled by the project defaults.

## Validation status (2026-10-19)

The GPU results below apply to the earlier transition-local-memory model.
The 2026-10-20 persistent-trace revision has separate CPU numerical, JIT,
joint-update, and checkpoint tests in VALIDATION.md. Its full GPU training
throughput and memory use have not yet been measured. Use a new log directory;
older NRSM checkpoints are rejected by the architecture schema check.

The isolated environment was installed successfully and `pip check` reported no broken requirements. The host was NVIDIA GeForce RTX 5090 D v2, driver 610.88, 24,455 MiB reported device memory.

`scripts/gpu_check.py --require-gpu` returned `GPU_CHECK_OK` on `cuda:0`. Float32 matrix multiplication, convolution and backpropagation passed; the first convolution compilation/execution took 0.417 seconds in this small probe. TFP categorical entropy and the complete `recall2imagine.Agent` import also passed. No JAX compatibility monkey patch was required.

The bounded integration check uses:

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.03 \
  "$HOME/.venvs/ctwm-r2i-gpu/bin/python" -m pytest tests/test_integration.py -q \
  -k 'native_discrete_and_continuous_joint_updates_and_restore or image_encoder_decoder_loss'
```

This command completed successfully: **2 passed, 6 deselected, 2 subtests passed in 391.21 seconds**. It covered image encoder/decoder loss; discrete and continuous model/actor/critic joint updates; world-model parameter isolation during actor training; and a serialized parameter restore round trip. The default XLA autotuning was enabled. Sampled total device use was 996–1,008 MiB while the background baseline was 226 MiB. This was a small-batch check, not a full 4×1024 training batch or a learning-performance benchmark.

The test emitted nonfatal deprecation warnings about TFP's old JAX array registration and JAX's int64-to-int32 conversion with x64 disabled. No compatibility patch or mixed precision was introduced to make it pass.

## Primary references

- [JAX installation](https://docs.jax.dev/en/latest/installation.html): CUDA pip wheels, WSL/Linux platform support and library search paths.
- [JAX release notes](https://github.com/jax-ml/jax/blob/main/CHANGELOG.md): NumPy requirements for JAX 0.6.2 and 0.7.2.
- [TFP maintainers' compatibility guidance](https://github.com/tensorflow/probability/issues/1994): use a date-pinned TFP nightly with newer JAX; stable TFP 0.25 targets JAX 0.4.35.
