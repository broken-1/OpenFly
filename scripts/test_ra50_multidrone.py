import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import ra50_multidrone as runner


class MultiDroneTests(unittest.TestCase):
    def test_pose_and_metric_mapping(self):
        self.assertEqual(runner.move([0,0,0,0],8),[6.,0.,0.,0.])
        self.assertEqual(runner.move([0,0,0,0],9),[9.,0.,0.,0.])
        self.assertAlmostEqual(runner.move([0,0,0,0],2)[3],np.pi/6)

    def test_settings_have_identical_origins_and_four_named_cameras(self):
        settings=runner.simulator_settings({'runtime':{'airsim_port':42451}})
        self.assertEqual(len(settings['Vehicles']),4)
        self.assertEqual(settings['PhysicsEngineName'],'ExternalPhysicsEngine')
        for vehicle in settings['Vehicles'].values():
            self.assertEqual([vehicle[k] for k in ('X','Y','Z')],[0,0,0])
            self.assertEqual(vehicle['Cameras']['front_custom']['CaptureSettings'][0]['Width'],1920)

    def test_concurrent_requests_and_per_task_history_reset(self):
        lock=threading.Lock()
        active=peak=0
        calls=[]
        class FakeDrone:
            def __init__(self,name,port):self.name=name
            def place(self,pose,pitch):self.pose=list(pose)
            def image(self):return np.zeros((1,1,3),dtype=np.uint8),.01
        class FakePolicy:
            def __init__(self,config):pass
            def decide(self,instruction,images,actions,seed):
                nonlocal active,peak
                with lock:
                    active+=1;peak=max(peak,active)
                    calls.append((seed,list(actions),len(images)))
                time.sleep(.02)
                with lock:active-=1
                return (1 if not actions else 0), {'raw_output':'{}','reason':'test','usage':{},
                    'preprocess_s':.01,'model_request_s':.02,'first_token_s':.01}
        config={'datasets':['configs/seen_airsim_23_random50_seed20260609.json'],
            'runtime':{'airsim_port':42451},'max_steps':5,'history_images':3,'seed':123}
        with tempfile.TemporaryDirectory() as d, patch.object(runner,'Drone',FakeDrone), patch.object(runner,'JsonPolicy',FakePolicy):
            report=runner.batch(config,Path(d),4,12,'unit')
            records=[json.loads(l) for l in (Path(d)/'unit_c4.jsonl').read_text().splitlines()]
        summaries=[r for r in records if r['record_type']=='sample_summary']
        self.assertEqual(len(summaries),12)
        self.assertEqual({r['sample_idx'] for r in summaries},set(range(12)))
        self.assertEqual(report['mean_steps'],2)
        self.assertGreaterEqual(peak,3)
        for seed in range(123,135):
            self.assertEqual([(a,n) for s,a,n in calls if s==seed],[([],1),([1],2)])


if __name__=='__main__':unittest.main()
