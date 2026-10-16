"""Visible session budget and measured training ETA, without changing execution."""
from datetime import datetime
import json
from pathlib import Path
import statistics
import threading
import time


def duration(seconds):
    minutes=max(0,int(seconds)//60)
    return f'{minutes//1440}d {(minutes%1440)//60}h {minutes%60}m' if minutes>=1440 else f'{minutes//60}h {minutes%60}m'


def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def training_estimate(records,measurement_id,step,target):
    # Never use source-machine timings or a partially resumed first step.
    seconds=[r['seconds'] for r in records if r.get('measurement_id')==measurement_id and r.get('seconds',0)>0][-5:]
    remaining=max(0,target-step)
    if not seconds or not remaining:return None
    return dict(samples=len(seconds),seconds_per_step=statistics.mean(seconds),
        remaining_seconds=statistics.mean(seconds)*remaining)


class RunProgress:
    def __init__(self,run,seed,hours,margin_minutes,measurement_id,emit,interval=60):
        self.run=Path(run);self.seed=seed;self.measurement_id=measurement_id;self.emit=emit
        self.start=time.time();self.end=self.start+hours*3600
        self.cutoff=self.end-margin_minutes*60;self.interval=interval
        self.stop=threading.Event();self.thread=None

    def report(self):
        now=time.time();folder=self.run/f'seed_{self.seed}'
        status=read_json(self.run/'status.json')
        commit=read_json(folder/'latest_checkpoint/commit.json')
        if not commit:commit=read_json(folder/'state.json')
        step=commit.get('step',0);target=120
        lines=[f'[run] Elapsed {duration(now-self.start)}; committed steps {step}/{target}; phase: {status.get("phase",status.get("status","starting"))}.']
        if now<self.cutoff:
            lines.append(f'[run] New-phase cutoff in {duration(self.cutoff-now)}; session budget remaining {duration(self.end-now)}.')
        else:lines.append(f'[run] Pause margin reached; waiting for the active phase and checkpoint save. Session budget remaining {duration(self.end-now)}.')
        records=[]
        try:
            for line in (folder/'step_timings.jsonl').read_text().splitlines():
                try:records.append(json.loads(line))
                except ValueError:continue
        except OSError:pass
        estimate=training_estimate(records,self.measurement_id,step,target)
        if estimate:
            remaining=estimate['remaining_seconds'];finish=datetime.fromtimestamp(now+remaining).astimezone().strftime('%Y-%m-%d %H:%M %Z')
            lines.append(f'[run] Training ETA: ~{duration(remaining)} of active compute (~{finish} if uninterrupted), based on {estimate["samples"]} complete step(s) in this invocation. Provisional; current-step work is counted conservatively.')
            lines.append('[run] Training ETA excludes validation, final/confirmation evaluations, runtime downtime and future setup. Full experiment finish: not yet estimated.')
            if now+remaining>self.end:lines.append('[run] Training is projected to exceed this session budget; additional resumed sessions are required.')
        elif step>=target:lines.append('[run] Training target reached. Evaluation/analysis may still be running; full finish ETA is not yet calibrated.')
        else:lines.append('[run] Training ETA: calibrating; waiting for a complete sampling + optimization + checkpoint cycle in this invocation. Resumed partial cycles are excluded.')
        for line in lines:self.emit(line+'\n')

    def __enter__(self):
        cutoff=datetime.fromtimestamp(self.cutoff).astimezone().strftime('%Y-%m-%d %H:%M %Z')
        end=datetime.fromtimestamp(self.end).astimezone().strftime('%Y-%m-%d %H:%M %Z')
        self.emit(f'[run] Planned new-phase cutoff: {cutoff}; session budget deadline: {end}. These are session limits, not a full experiment completion promise; checkpoint saving can take additional time.\n')
        self.report()
        def heartbeat():
            while not self.stop.wait(self.interval):self.report()
        self.thread=threading.Thread(target=heartbeat,daemon=True);self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set();self.thread.join()
