"""CT-WM entry point; implementation and upstream infrastructure live in packages."""
import sys
import os
from pathlib import Path


def main():
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(
        Path(__file__).resolve().parent / 'outputs' / '.jax_cache'))
    argv = sys.argv[1:]
    if not argv:
        print('Short check: python CTWM.py --configs ctwm ctwm_smoke '
              '--task bsuite_memory_len/4 --logdir outputs/smoke --run.steps 200')
        print('See README.md for installation, GPU runs, and full-size settings.')
        return
    from recall2imagine.train import main as train
    if not any(x == '--configs' or x.startswith('--configs=') for x in argv):
        argv = ['--configs', 'ctwm', *argv]
    train(argv)


if __name__ == '__main__':
    main()
