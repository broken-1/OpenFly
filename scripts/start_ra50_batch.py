"""Own a benchmark model service and close it when the evaluation exits."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    config=json.loads(args.config.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',8004))==0:
            raise RuntimeError('Port 8004 occupied; existing service will not be touched')
    subprocess.run([sys.executable,str(root/'scripts/record_experiment.py'),
        '--config',str(args.config),'--output',str(args.output/'run_manifest.json')],check=True)
    env=os.environ.copy()
    env.pop('PYTHONPATH',None)
    env.update(config['runtime']['server_env'])
    env['PATH']='/home/liusongbo/.venvs/vllm-qwen27b/bin:'+env.get('PATH','')
    child=None
    def interrupted(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    with (args.output/'model.log').open('a') as log:
        model=subprocess.Popen(config['runtime']['server_command'],env=env,cwd=root,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        (args.output/'model.pid').write_text(str(model.pid))
        try:
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for _ in range(150):
                if model.poll() is not None:
                    raise RuntimeError('Model startup failed; inspect model.log')
                try:
                    with opener.open('http://127.0.0.1:8004/health',timeout=2) as response:
                        if response.status==200:
                            break
                except OSError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError('Model did not become ready')
            child=subprocess.Popen([sys.executable,str(root/'scripts/ra50_multidrone.py'),
                '--config',str(args.config),'--output',str(args.output)],cwd=root)
            code=child.wait()
            if code:
                raise SystemExit(code)
        finally:
            if child and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if model.poll() is None:
                os.killpg(model.pid,signal.SIGTERM)
                try:
                    model.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(model.pid,signal.SIGKILL)
                    model.wait()


if __name__=='__main__':
    main()
