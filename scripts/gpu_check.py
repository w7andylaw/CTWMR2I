"""Bounded float32 JAX check; never starts training or accepts a CPU fallback silently."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()
    # Set before importing JAX. 3% of the tested 24 GiB GPU is under 1 GiB.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.03')
    os.environ.setdefault('JAX_ENABLE_X64', 'false')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    report = {'python': sys.executable, 'float_dtype': 'float32'}
    for name in ('jax', 'jaxlib', 'tfp-nightly', 'tensorflow-probability', 'flax', 'optax', 'numpy'):
        try:
            report[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        driver = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.used,memory.total',
             '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        report['nvidia_smi'] = (driver.stdout or driver.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        report['nvidia_smi'] = str(exc)
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        jax.config.update('jax_default_matmul_precision', 'highest')
        devices = jax.devices()
        report['devices'] = [str(device) for device in devices]
        gpus = [device for device in devices if device.platform == 'gpu']
        if args.require_gpu and not gpus:
            raise RuntimeError('JAX did not register a GPU; CPU fallback is disabled.')
        device = gpus[0] if gpus else devices[0]
        with jax.default_device(device):
            x = jnp.ones((128, 128), jnp.float32)
            matmul = jax.jit(lambda a: a @ a)(x)
            matmul.block_until_ready()
            np.testing.assert_allclose(np.asarray(matmul), 128.0)
            image = jnp.ones((1, 16, 16, 3), jnp.float32)
            kernel = jnp.ones((3, 3, 3, 4), jnp.float32)

            def objective(weight):
                conv = jax.lax.conv_general_dilated(
                    image, weight, (1, 1), 'VALID',
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
                return jnp.square(conv).mean()

            start = time.perf_counter()
            loss, gradient = jax.jit(jax.value_and_grad(objective))(kernel)
            gradient.block_until_ready()
            np.testing.assert_allclose(float(loss), 729.0)
            assert np.isfinite(np.asarray(gradient)).all()
            assert gradient.dtype == jnp.float32
            report['conv_compile_and_run_seconds'] = time.perf_counter() - start
            report['probe_device'] = str(matmul.device)
            report['probe_dtype'] = str(gradient.dtype)
            # Trigger the same TFP distributions used by the agent, not just a lazy import.
            from tensorflow_probability.substrates import jax as tfp
            distribution = tfp.distributions.OneHotCategorical(logits=jnp.zeros((2, 4)))
            report['tfp_entropy'] = np.asarray(distribution.entropy()).tolist()
            import recall2imagine
            report['agent_import'] = recall2imagine.Agent.__name__
        report['status'] = 'GPU_CHECK_OK' if gpus else 'CPU_CHECK_OK'
        code = 0
    except Exception as exc:
        report.update(status='GPU_CHECK_FAILED', error=f'{type(exc).__name__}: {exc}')
        code = 1
    result = json.dumps(report, ensure_ascii=False, indent=2)
    print(result, flush=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(result + '\n', encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
