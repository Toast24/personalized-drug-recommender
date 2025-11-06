import subprocess
import sys

# Quick smoke test: run the train script for 1 epoch in demo mode
# This test is intended to be run manually; here we provide a quick helper.

if __name__ == '__main__':
    cmd = [sys.executable, '-m', 'src.train', '--mode', 'demo', '--epochs', '1', '--n-patients', '100']
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)
