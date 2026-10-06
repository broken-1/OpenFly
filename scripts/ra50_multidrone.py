"""Independent RA50 tasks in one AirSim scene, sharing one local vLLM server."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import statistics
import subprocess
import threading
import time

import numpy as np
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
INTERRUPTED = threading.Event()
ACTIONS = {0: "STOP", 1: "FORWARD_3", 2: "TURN_LEFT_30", 3: "TURN_RIGHT_30",
           4: "UP_3", 5: "DOWN_3", 6: "LEFT_3", 7: "RIGHT_3",
           8: "FORWARD_6", 9: "FORWARD_9"}
IDS = {name: idx for idx, name in ACTIONS.items()}


def move(pose, action):
    x, y, z, yaw = pose
    if action in (1, 8, 9):
        distance = {1: 3, 8: 6, 9: 9}[action]
        x += distance * math.cos(yaw)
        y += distance * math.sin(yaw)
    elif action == 2:
        yaw += math.pi / 6
    elif action == 3:
        yaw -= math.pi / 6
    elif action == 4:
        z += 3
    elif action == 5:
        z -= 3
    elif action == 6:
        x -= 3 * math.sin(yaw)
        y += 3 * math.cos(yaw)
    elif action == 7:
        x += 3 * math.sin(yaw)
        y -= 3 * math.cos(yaw)
    elif action != 0:
        raise ValueError(action)
    return [x, y, z, (yaw + math.pi) % (2 * math.pi) - math.pi]


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a[:3], b[:3])))


class JsonPolicy:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.trust_env = False
        self.prompt = (ROOT / config['prompt']['path']).read_text()

    def decide(self, instruction, images, actions, seed):
        prepared = time.perf_counter()
        content = []
        for i, array in enumerate(images):
            data = io.BytesIO()
            Image.fromarray(array).save(data, format='PNG')
            content.extend([
                {'type': 'text', 'text': f'Observation {i+1}: {"current" if i==len(images)-1 else "previous"} view.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()}}])
        content.append({'type': 'text', 'text': self.prompt.format(
            instruction=instruction, actions=json.dumps([ACTIONS[a] for a in actions[-10:]]))})
        preprocess_s = time.perf_counter() - prepared
        schema = {'type': 'object', 'properties': {
            'action': {'type': 'string', 'enum': list(IDS)}, 'reason': {'type': 'string'}},
            'required': ['action', 'reason'], 'additionalProperties': False}
        start = time.perf_counter()
        first = None
        text = ''
        usage = None
        finish = None
        with self.session.post(self.config['runtime']['base_url']+'/chat/completions',
            json={'model': self.config['runtime']['served_model'],
                  'messages': [{'role': 'user', 'content': content}],
                  'temperature': 0, 'top_p': 1, 'top_k': -1, 'seed': seed,
                  'max_tokens': self.config['runtime']['max_output_tokens'],
                  'chat_template_kwargs': {'enable_thinking': False},
                  'structured_outputs': {'json': schema},
                  'stream': True, 'stream_options': {'include_usage': True}},
            stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b'data: '):
                    continue
                if line == b'data: [DONE]':
                    break
                chunk = json.loads(line[6:])
                if 'error' in chunk:
                    raise RuntimeError(chunk['error'])
                if chunk.get('usage'):
                    usage = chunk['usage']
                if chunk.get('choices'):
                    choice = chunk['choices'][0]
                    piece = choice['delta'].get('content') or ''
                    if piece and first is None:
                        first = time.perf_counter()
                    text += piece
                    finish = choice.get('finish_reason') or finish
        if finish != 'stop':
            raise RuntimeError(f'Incomplete JSON generation ({finish}): {text}')
        result = json.loads(text)
        if result.get('action') not in IDS or not isinstance(result.get('reason'), str):
            raise ValueError(f'Invalid action JSON: {text}')
        return IDS[result['action']], {
            'raw_output': text, 'reason': result['reason'], 'usage': usage,
            'preprocess_s': preprocess_s, 'model_request_s': time.perf_counter()-start,
            'first_token_s': first-start if first else None,
        }


def simulator_settings(config):
    base = json.loads((ROOT / 'envs/airsim/AirSim/settings.json').read_text())
    prototype = copy.deepcopy(base['Vehicles']['drone_1'])
    prototype['Sensors'] = {}  # Evaluation uses RGB only; identical in all groups.
    prototype.update(X=0, Y=0, Z=0, Yaw=0, EnableCollisions=False)
    prototype['Cameras']['front_custom']['CaptureSettings'] = [
        entry for entry in prototype['Cameras']['front_custom']['CaptureSettings']
        if entry['ImageType'] == 0]
    base['Vehicles'] = {f'drone_{i+1}': copy.deepcopy(prototype) for i in range(4)}
    base.update(ApiServerPort=config['runtime']['airsim_port'], RpcEnabled=True,
                ViewMode='NoDisplay', EngineSound=False,
                PhysicsEngineName='ExternalPhysicsEngine')
    base['Recording']['Enabled'] = False
    return base


class Drone:
    def __init__(self, name, port):
        import airsim
        self.airsim = airsim
        self.name = name
        self.client = airsim.MultirotorClient(ip='127.0.0.1', port=port, timeout_value=30)

    def place(self, pose, pitch):
        a = self.airsim
        orientation = a.to_quaternion(math.radians(pitch), 0, -pose[3])
        self.client.simSetVehiclePose(a.Pose(a.Vector3r(pose[0], -pose[1], -pose[2]),
            orientation),
            ignore_collision=True, vehicle_name=self.name)
        deadline = time.monotonic() + 3
        while True:
            observed = self.client.simGetVehiclePose(vehicle_name=self.name)
            actual = observed.position
            measured = [actual.x_val, -actual.y_val, -actual.z_val]
            alignment = abs(sum(getattr(orientation, k)*getattr(observed.orientation, k)
                                for k in ('x_val','y_val','z_val','w_val')))
            if distance(pose, measured) <= .1 and alignment >= .99999:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Vehicle {self.name} pose mismatch: expected {pose[:3]}, got {measured}')
            time.sleep(.02)

    def image(self):
        started = time.perf_counter()
        time.sleep(.15)  # Allow the render frame to reflect the new vehicle pose.
        a = self.airsim
        response = self.client.simGetImages([
            a.ImageRequest('front_custom', a.ImageType.Scene, False, False)],
            vehicle_name=self.name)[0]
        if response.width != 1920 or response.height != 1080:
            raise RuntimeError(f'Unexpected camera size {response.width}x{response.height}')
        image = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(
            response.height, response.width, 3).copy()
        return image, time.perf_counter()-started


@contextmanager
def deployed_settings(settings, output):
    """This packaged scene reads its binary-adjacent settings before CLI overrides."""
    target = ROOT/'envs/airsim/env_airsim_23/LinuxNoEditor/AirVLN/Binaries/Linux/settings.json'
    with open('/tmp/openfly_ra50_env23.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',41451))==0:
                raise RuntimeError('Existing default AirSim server; settings will not be changed')
        original = target.read_bytes()
        replacement = settings.read_bytes()
        (output/'original_binary_settings.json').write_bytes(original)
        target.write_bytes(replacement)
        try:
            yield
        finally:
            if target.read_bytes()==replacement:
                target.write_bytes(original)
            else:
                print('Binary settings changed externally; preserved current file and saved original backup',flush=True)


def batch(config, output, concurrency, limit, phase):
    tasks = json.loads((ROOT / config['datasets'][0]).read_text())[:limit]
    if not tasks or any(not t['image_path'].startswith('env_airsim_23/') for t in tasks):
        raise ValueError('This test requires nonempty single-scene RA50 tasks')
    jobs = queue.Queue()
    for idx, item in enumerate(tasks):
        jobs.put((idx, item))
    lock = threading.Lock()
    abort = INTERRUPTED
    ready = threading.Barrier(concurrency+1)
    summaries = []
    destination = output / f'{phase}_c{concurrency}.jsonl'
    if destination.exists():
        raise FileExistsError(destination)
    destination.touch()

    def log(record):
        with lock:
            with destination.open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False)+'\n')

    def worker(worker_id):
        drone = Drone(f'drone_{worker_id+1}', config['runtime']['airsim_port'])
        policy = JsonPolicy(config)
        ready.wait(timeout=60)
        while not abort.is_set():
            try:
                idx, item = jobs.get_nowait()
            except queue.Empty:
                break
            began = time.perf_counter()
            pose = [*item['pos'][0], item['yaw'][0]]
            goal = item['pos'][-1]
            pitch = -45 if 'high' in item['image_path'] else 0
            images, actions, timings = [], [], []
            traveled = 0.
            minimum = distance(pose, goal)
            try:
                for step in range(config['max_steps']):
                    if abort.is_set():
                        return
                    started = time.perf_counter()
                    drone.place(pose, pitch)
                    image, capture_s = drone.image()
                    images = (images+[image])[-config['history_images']:]
                    action, details = policy.decide(item['gpt_instruction'], images,
                        actions, config['seed']+idx)
                    timings.append({'capture_s': capture_s, **{k: details[k] for k in
                        ('preprocess_s', 'model_request_s', 'first_token_s')}})
                    log({'record_type': 'step', 'sample_idx': idx, 'worker_id': worker_id,
                         'vehicle_name': drone.name, 'step': step, 'pose': pose,
                         'action': action, 'action_name': ACTIONS[action],
                         'elapsed_s': time.perf_counter()-started, **details,
                         'capture_s': capture_s})
                    actions.append(action)
                    if action == 0:
                        break
                    previous = pose
                    pose = move(pose, action)
                    traveled += distance(previous, pose)
                    minimum = min(minimum, distance(pose, goal))
                final = distance(pose, goal)
                shortest = distance(item['pos'][0], goal)
                summary = {'record_type': 'sample_summary', 'sample_idx': idx,
                    'vehicle_name': drone.name, 'steps': len(actions), 'actions': actions,
                    'stopped': actions[-1]==0, 'NE': final, 'SR': int(final<20),
                    'OSR': int(minimum<20),
                    'SPL': shortest/max(traveled, shortest, 1e-12) if final<20 else 0.,
                    'elapsed_s': time.perf_counter()-began, 'final_pose': pose,
                    'mean_capture_s': statistics.mean(t['capture_s'] for t in timings),
                    'mean_preprocess_s': statistics.mean(t['preprocess_s'] for t in timings),
                    'mean_model_request_s': statistics.mean(t['model_request_s'] for t in timings)}
                log(summary)
                with lock:
                    summaries.append(summary)
                print(f'{phase} c={concurrency} sample={idx} steps={len(actions)} '
                      f'SR={summary["SR"]} seconds={summary["elapsed_s"]:.1f}', flush=True)
            except BaseException as error:
                abort.set()
                log({'record_type': 'error', 'sample_idx': idx, 'error': repr(error)})
                raise
            finally:
                drone.place([-4000-worker_id*100, 5000, 2000, 0], 0)
                jobs.task_done()

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i) for i in range(concurrency)]
        ready.wait(timeout=60)
        # Includes all task work and scheduling, excludes simulator/model startup.
        start = time.perf_counter()
        for future in futures:
            future.result()
    elapsed = time.perf_counter()-start
    if {s['sample_idx'] for s in summaries} != set(range(len(tasks))):
        raise RuntimeError('Incomplete or duplicate task batch')
    count = len(summaries)
    report = {'phase': phase, 'concurrency': concurrency, 'samples': count,
        'wall_s': elapsed, 'tasks_per_hour': count*3600/elapsed,
        'actions_per_second': sum(s['steps'] for s in summaries)/elapsed,
        'mean_steps': statistics.mean(s['steps'] for s in summaries),
        'mean_request_s': sum(s['mean_model_request_s']*s['steps'] for s in summaries)/sum(s['steps'] for s in summaries),
        'NE': statistics.mean(s['NE'] for s in summaries),
        **{k+'_pct':100*statistics.mean(s[k] for s in summaries) for k in ('SR','OSR','SPL')}}
    (output/f'{phase}_c{concurrency}_summary.json').write_text(json.dumps(report,indent=2))
    return report


def run(config, output):
    import airsim
    output.mkdir(parents=True, exist_ok=True)
    settings = output/'airsim_settings.json'
    settings.write_text(json.dumps(simulator_settings(config), indent=2))
    port = config['runtime']['airsim_port']
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:
            raise RuntimeError('AirSim RPC port already in use; no existing process touched')
    simulator_gpu = config['runtime'].get('simulator_gpu', config['runtime']['gpu'])
    command = ['bash', str(ROOT/'envs/airsim/env_airsim_23/LinuxNoEditor/start.sh'),
               '-RenderOffscreen', '-NoSound', '-vulkan', f'-graphicsadapter={simulator_gpu}',
               '-settings='+str(settings.resolve())]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=str(simulator_gpu)
    reports = []
    with deployed_settings(settings, output), (output/'simulator.log').open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        (output/'simulator.pid').write_text(str(process.pid))
        try:
            client = None
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError('Simulator exited; inspect simulator.log')
                try:
                    trial = airsim.MultirotorClient(ip='127.0.0.1', port=port, timeout_value=2)
                    if trial.ping():
                        client=trial
                        break
                except Exception:
                    pass
                time.sleep(2)
            if client is None:
                raise TimeoutError('AirSim did not become ready')
            if set(client.listVehicles()) != {f'drone_{i+1}' for i in range(4)}:
                raise RuntimeError('Simulator did not load the four-vehicle settings')
            # External physics keeps poses fixed without gravity, while ticks
            # continue to apply vehicle poses and update camera rendering.
            client.simPause(False)
            for i in range(4):
                drone = Drone(f'drone_{i+1}', port)
                drone.place([-4000-i*100,5000,2000,0],0)
            first = json.loads((ROOT/config['datasets'][0]).read_text())[0]
            probe = Drone('drone_1', port)
            first_pose = [*first['pos'][0], first['yaw'][0]]
            probe.place(first_pose, -45 if 'high' in first['image_path'] else 0)
            initial, _ = probe.image()
            probe.place(move(first_pose,9), -45 if 'high' in first['image_path'] else 0)
            shifted, _ = probe.image()
            if np.array_equal(initial, shifted):
                raise RuntimeError('Camera did not update after changing the vehicle pose')
            Image.fromarray(initial).save(output/'preflight_start.jpg')
            Image.fromarray(shifted).save(output/'preflight_forward9.jpg')
            probe.place([-4000,5000,2000,0],0)
            for concurrency in config['concurrency']:
                # Same excluded warmup before each measured group; caches are
                # disabled so previous groups cannot serve repeated tasks cheaply.
                def warmup(_):
                    return JsonPolicy(config).decide(first['gpt_instruction'],
                        [initial]*config['history_images'], [], config['seed'])
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(warmup,range(4)))
                report=batch(config, output, concurrency, config['sample_limit'], 'ra50')
                reports.append(report)
                if len(reports)>1:
                    report['task_throughput_speedup']=report['tasks_per_hour']/reports[0]['tasks_per_hour']
                (output/'comparison.json').write_text(json.dumps(reports,indent=2))
                print('BATCH SUMMARY',json.dumps(report),flush=True)
            return reports
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    def interrupted(signum, frame):
        INTERRUPTED.set()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    config=json.loads(args.config.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    status=args.output/'status.json'
    status.write_text(json.dumps({'status':'running'}))
    try:
        run(config,args.output)
        status.write_text(json.dumps({'status':'completed'}))
    except BaseException as error:
        status.write_text(json.dumps({'status':'failed','error':repr(error)}))
        raise


if __name__=='__main__':
    main()
