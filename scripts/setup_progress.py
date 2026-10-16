"""Stream setup subprocesses into the cell and a timed local log."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from session_store import atomic_json


class SetupReporter:
    def __init__(self,folder,heartbeat_seconds=20):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.interval=heartbeat_seconds;self.lock=threading.RLock()

    def output(self,message):
        with self.lock:
            print(message,end='' if message.endswith('\n') else '\n',flush=True)
            with (self.folder/'setup.log').open('a') as stream:
                stream.write(message);stream.write('' if message.endswith('\n') else '\n')

    @contextmanager
    def stage(self,name):
        start=time.monotonic();stop=threading.Event()
        def status(value):
            with self.lock:
                atomic_json(self.folder/'setup_status.json',dict(stage=name,status=value,
                    elapsed_seconds=round(time.monotonic()-start,1),updated_at=time.time()))
        def heartbeat():
            while not stop.wait(self.interval):
                status('RUNNING')
                self.output(f'[setup] {name}: still running ({time.monotonic()-start:.0f}s elapsed)')
        self.output(f'[setup] START {name}');status('RUNNING')
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        result='COMPLETED'
        try:yield
        except BaseException as error:
            result='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED'
            raise
        finally:
            stop.set();thread.join();status(result)
            self.output(f'[setup] {result} {name} ({time.monotonic()-start:.1f}s)')

    def run(self,name,command,env=None):
        with self.stage(name):
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,env=env,start_new_session=True)
            try:
                for line in process.stdout:self.output(line)
                code=process.wait()
                if code:raise subprocess.CalledProcessError(code,command)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
            finally:process.stdout.close()


def setup_process_kind(arguments,project):
    """Match only this checkout's setup scripts, never a training driver."""
    for index,argument in enumerate(arguments):
        if argument==str(project/'scripts/bootstrap_notebook.py'):return 'installer'
        if argument==str(project/'scripts/artifacts.py'):
            if 'restore' in arguments[index+1:]:return 'restore'
            if 'download' in arguments[index+1:]:return 'download'
        if argument==str(project/'studies/rlvr_success_transfer/reproduce_failure.py'):
            if index+1<len(arguments) and arguments[index+1]=='prepare':return 'prepare'
    return None


def stop_previous_setup(project):
    """Explicit --restart stops leftover installers, retaining on-disk caches."""
    processes={};roots=[]
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            if directory.stat().st_uid!=os.getuid():continue
            pid=int(directory.name)
            if pid==os.getpid():continue
            args=directory.joinpath('cmdline').read_bytes().decode().split('\0')
            stat=directory.joinpath('stat').read_text().split(') ')[1].split()
            processes[pid]=(int(stat[1]),stat[19])
            kind=setup_process_kind(args,project)
            if kind=='restore':raise RuntimeError(f'Checkpoint restore PID {pid} is still active. Let it finish, then rerun setup; it will not be killed.')
            if kind:roots.append(pid)
        except (OSError,ValueError,IndexError,UnicodeDecodeError):continue
    targets=set(roots)
    while True:
        expanded=targets|{pid for pid,(parent,_) in processes.items() if parent in targets}
        if expanded==targets:break
        targets=expanded
    for signum in [signal.SIGTERM,signal.SIGKILL]:
        for pid in targets:
            try:
                current=Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()
                if current[19]==processes[pid][1]:os.kill(pid,signum)
            except (OSError,IndexError):pass
        if signum==signal.SIGTERM and targets:time.sleep(1)
    return sorted(targets)
