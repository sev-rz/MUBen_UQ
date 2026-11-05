"""Operational wrapper around tierb_chemberta.worker (scientific code untouched, tierb_chemberta.py digest unchanged).
A unit that hits CUDA out-of-memory (another process on the GPU) is retried from its seed after the cache is freed, instead of
being recorded as failed. A restarted unit re-seeds and retrains, so results equal an uninterrupted run.
  python cb_robust_worker.py [--splits ...] [--datasets ...]   (default: all splits, all datasets)"""
import sys, gc, time, argparse
import torch
import tierb_chemberta as cb
import tierb_dnn as t

TRIES, WAIT = 20, 90
_orig = cb.run_unit

def run_unit(c, kind, seed):
    for a in range(TRIES):
        try:
            return _orig(c, kind, seed)
        except torch.OutOfMemoryError as ex:
            print('OOM_RETRY', c.split, c.d, kind, seed, a + 1, str(ex)[:150].replace('\n', ' '), flush=True)
        gc.collect(); torch.cuda.empty_cache(); time.sleep(WAIT)
    return _orig(c, kind, seed)      # last attempt: any error reaches the worker's failure record

cb.run_unit = run_unit

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--splits', nargs='*', default=t.SPLITS); p.add_argument('--datasets', nargs='*', default=t.DATASETS); a = p.parse_args()
    cb.worker(a.splits, a.datasets)
