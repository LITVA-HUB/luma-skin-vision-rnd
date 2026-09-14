"""Finish this one ML workflow after the already-running training worker exits successfully."""
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from skin_data_growth import DATA_ROOT, sha_bytes, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = DATA_ROOT/'facial_skin_v1'


def wait_for_worker(pid):
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong,ctypes.c_bool,ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000 | 0x100000,False,pid)
    if not handle:
        raise RuntimeError('cannot attach to the expected active training worker')
    started = time.monotonic()
    try:
        while True:
            state = kernel.WaitForSingleObject(handle,1000)
            if state == 0:
                code = ctypes.c_ulong()
                if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value != 0:
                    raise RuntimeError(f'training worker did not exit successfully: {code.value}')
                return
            if state != 0x102 or time.monotonic()-started > 3*3600:
                raise RuntimeError('training wait failed or exceeded three hours')
    finally:
        kernel.CloseHandle(handle)


def main():
    job = json.loads((OUT/'job.json').read_text())
    if job['stage'] != 'training':
        raise RuntimeError('expected the current training stage')
    state = dict(stage='waiting_for_training',pipeline_pid=os.getpid(),training_pid=job['pid'],
                 pipeline_source_sha256=sha_bytes(Path(__file__).read_bytes()),
                 protocol_sha256=sha_bytes((OUT/'protocol.json').read_bytes()))
    write_json(OUT/'pipeline.json',state)
    print('SEG PIPELINE waiting for successful training exit',job['pid'],flush=True)
    try:
        wait_for_worker(job['pid'])
        selection = json.loads((OUT/'selection.json').read_text())
        if selection['protocol_sha256'] != state['protocol_sha256']:
            raise ValueError('selection does not belong to the queued training run')
        for stage,script in [('evaluation','skin_face_evaluate.py'),('diagnostics','skin_face_diagnostic.py')]:
            state['stage'] = stage
            write_json(OUT/'pipeline.json',state)
            print('SEG PIPELINE',stage,flush=True)
            with (OUT/(stage+'.log')).open('w',encoding='utf-8') as log:
                proc = subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError(f'{stage} failed with exit code {proc.returncode}; see {stage}.log')
        state['stage'] = 'complete'
        state['test_results_sha256'] = sha_bytes((OUT/'test_results.json').read_bytes())
        state['diagnostic_sha256'] = sha_bytes((OUT/'diagnostic.json').read_bytes())
        write_json(OUT/'pipeline.json',state)
        print('SEG PIPELINE COMPLETE',json.dumps(state),flush=True)
    except Exception as exc:
        state.update(stage='failed',error=str(exc))
        write_json(OUT/'pipeline.json',state)
        raise


if __name__ == '__main__':
    main()
