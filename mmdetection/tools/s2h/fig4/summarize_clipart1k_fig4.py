"""Date-scoped Figure 4 summary wrapper.

Keeps Figure 4 artifacts separate from the general S2H analysis outputs while
delegating aggregation and plotting to the existing, tested sweep summarizer.
"""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='20261002')
    parser.add_argument('--root', default=None)
    parser.add_argument('--seeds', nargs='+', default=['3407', '3408', '3409'])
    parser.add_argument('--params', nargs='+',
                        default=['g_max', 'theta', 'con_weight',
                                 'ffcp_rank', 'hard_radius'])
    args = parser.parse_args()
    root = args.root or f'cat_work_dir/fig4/{args.tag}'
    out = f'work_dirs/s2h_analysis/fig4/{args.tag}/clipart1k_5shot'
    script = os.path.join(os.path.dirname(__file__), '..', 'analysis',
                          'summarize_sweep.py')
    command = [sys.executable, script, '--root', root, '--dataset', 'clipart1k',
               '--shot', '5', '--params', *args.params, '--seeds', *args.seeds,
               '--out', out]
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
