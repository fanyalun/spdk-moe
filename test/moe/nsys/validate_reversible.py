#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3


def main(directory):
    pattern = re.compile(r'^(\w+) expert=(-?\d+) stage=(-?\d+) bytes=(\d+)$')
    with sqlite3.connect(directory / 'lattice.sqlite') as connection:
        data = connection.execute('SELECT eventType, start, end, text, globalTid '
                                  'FROM NVTX_EVENTS ORDER BY start').fetchall()
        thread_schema = connection.execute('PRAGMA table_info(ThreadNames)').fetchall()
        threads = connection.execute('SELECT * FROM ThreadNames').fetchall()
        strings = dict(connection.execute('SELECT id, value FROM StringIds').fetchall())
    events = []
    for event_type, start, end, text, tid in data:
        match = pattern.match((text or '').split(' | meta:')[-1])
        if match:
            kind, expert, stage, size = match.groups()
            if kind != 'WAIT_WORKER' and (end is None or end <= start):
                raise ValueError(f'incomplete required range: {text}')
            events.append(dict(kind=kind, expert=int(expert), stage=int(stage), bytes=int(size),
                               start=start, end=end, tid=tid, event_type=event_type, text=text))
    counts = Counter(event['kind'] for event in events)
    config = json.loads((directory / 'lattice.json').read_text())
    params = next(entry['params'] for subsystem in config['subsystems']
                  for entry in subsystem['config'] if entry['method'] == 'bdev_moe_create')
    tiled = params.get('pipeline') == 'tile_fifo'
    requests = [event for event in events if event['kind'] == 'REQUEST']
    if len(requests) != 50 or [event['bytes'] for event in requests] != list(range(17, 67)):
        raise ValueError('report must contain exactly requests 17 through 66')
    for kind in ('ROUTE', 'ROUTER', 'TOPK', 'NORMALIZE', 'COMBINE'):
        if counts[kind] != 50:
            raise ValueError(f'{kind}: expected 50, got {counts[kind]}')
    for kind in ('GATE', 'UP', 'SILU_MULTIPLY', 'DOWN'):
        if counts[kind] < 400 or (not tiled and counts[kind] != 400):
            raise ValueError(f'{kind}: expected 400, got {counts[kind]}')
    thread_kinds = {}
    for kind in ('ROUTER', 'TOPK', 'NORMALIZE', 'GATE', 'UP', 'DOWN', 'COMBINE',
                 'IO_SUBMIT', 'DECODE'):
        group = [event for event in events if event['kind'] == kind]
        if not group or any(event['event_type'] != 59 for event in group):
            raise ValueError(f'{kind} must use native thread-local push/pop ranges')
        thread_kinds[kind] = sorted({event['tid'] for event in group})
    if set(thread_kinds['GATE']) & set(thread_kinds['DECODE']):
        raise ValueError('expected distinct compute and decode threads')
    for request in requests:
        contained = [event for event in events if event['start'] >= request['start'] and
                     event['end'] is not None and event['end'] <= request['end']]
        expert_sets = [{event['expert'] for event in contained if event['kind'] == kind}
                      for kind in ('GATE', 'UP', 'DOWN')]
        if any(len(group) != 8 or min(group) < 0 for group in expert_sets):
            raise ValueError('every request must compute eight identified experts')
        if expert_sets[0] != expert_sets[1] or expert_sets[0] != expert_sets[2]:
            raise ValueError('Gate, Up and Down expert IDs differ')
        if tiled:
            for kind in ('GATE', 'UP', 'DOWN'):
                expected = params.get('d_model', 2048) if kind == 'DOWN' else params.get('d_ff', 7168)
                for expert in expert_sets[0]:
                    group = [event for event in contained if event['kind'] == kind and
                             event['expert'] == expert]
                    ranges = []
                    for event in group:
                        match = re.search(r'channels=(\d+)\.\.(\d+)', event['text'])
                        ranges.append((int(match[1]), int(match[2]) + 1) if match else (0, expected))
                    cursor = 0
                    for first, end in sorted(ranges):
                        if first != cursor or end <= first:
                            raise ValueError('slice channel coverage has gaps or overlaps')
                        cursor = end
                    if cursor != expected:
                        raise ValueError('slice channel coverage incomplete')
    if counts['DECODE'] != counts['WEIGHT_IO'] or counts['DECODE'] != counts['IO_SUBMIT']:
        raise ValueError('read, submit and decode block counts differ')
    if not counts['WAIT_MATRIX'] or not counts['WAIT_WORKER']:
        raise ValueError('explicit pipeline and worker wait ranges missing')
    phase_ids = {tid for group in thread_kinds.values() for tid in group}
    thread_fields = [field[1] for field in thread_schema]
    names = []
    for values in threads:
        row = dict(zip(thread_fields, values))
        if row['globalTid'] in phase_ids:
            names.append(dict(globalTid=row['globalTid'], priority=row['priority'],
                              name=strings[row['nameId']]))
    summary = dict(passed=True, requests=50, counts=dict(counts), phase_threads=thread_kinds,
                   thread_names=names,
                   examples={kind: next(event['text'] for event in events if event['kind'] == kind)
                             for kind in thread_kinds},
                   gui_verified=False, note='Native report and exported events verified; GUI unavailable locally')
    (directory / 'validation.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    main(parser.parse_args().directory)
