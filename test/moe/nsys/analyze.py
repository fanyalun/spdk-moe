#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
from pathlib import Path
import re
import sqlite3


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError('invalid interval')
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection(left, right):
    result = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            result.append([start, end])
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return result


def analyze(directory, label):
    connection = sqlite3.connect(directory / f'{label}.sqlite')
    raw = connection.execute('SELECT start, end, text, globalTid FROM NVTX_EVENTS ORDER BY start').fetchall()
    events = []
    pattern = re.compile(r'^(REQUEST|COMPUTE|WEIGHT_IO|ROUTE|COMBINE) expert=(-?\d+) stage=(-?\d+) bytes=(\d+)$')
    for start, end, text, thread in raw:
        match = pattern.match((text or '').split(' | meta:')[-1])
        if match:
            if end is None:
                raise ValueError(f'incomplete NVTX range: {text}')
            kind, expert, stage, size = match.groups()
            events.append(dict(start=start, end=end, kind=kind, expert=int(expert),
                               stage=int(stage), size=int(size), thread=thread))
    requests = [event for event in events if event['kind'] == 'REQUEST']
    benchmark = json.loads((directory / f'{label}_bench.json').read_text())
    rows = []
    for index, request in enumerate(requests):
        contained = [event for event in events
                     if request['start'] <= event['start'] and event['end'] <= request['end']]
        compute = [event for event in contained if event['kind'] == 'COMPUTE']
        reads = [event for event in contained if event['kind'] == 'WEIGHT_IO']
        compute_union = merge((event['start'], event['end']) for event in compute)
        io_union = merge((event['start'], event['end']) for event in reads)
        overlap = duration(intersection(compute_union, io_union))
        same_expert = []
        for expert in {event['expert'] for event in compute}:
            expert_compute = merge((event['start'], event['end']) for event in compute
                                   if event['expert'] == expert)
            expert_io = merge((event['start'], event['end']) for event in reads
                              if event['expert'] == expert)
            same_expert.extend(intersection(expert_compute, expert_io))
        total = request['end'] - request['start']
        compute_ns, io_ns = duration(compute_union), duration(io_union)
        rows.append(dict(request=index + 1, start_seconds=request['start'] / 1e9,
                         total_ms=total / 1e6, compute_ms=compute_ns / 1e6,
                         io_inflight_ms=io_ns / 1e6, overlap_ms=overlap / 1e6,
                         same_expert_overlap_ms=duration(merge(same_expert)) / 1e6,
                         io_without_compute_ms=(io_ns - overlap) / 1e6,
                         neither_ms=(total - compute_ns - io_ns + overlap) / 1e6,
                         io_bytes=sum(event['size'] for event in reads),
                         io_outstanding_time_ms=sum(event['end'] - event['start'] for event in reads) / 1e6,
                         compute_ranges=len(compute), io_ranges=len(reads)))
    sample_count = benchmark['throughput']['total_calls']
    if len(rows) != sample_count + 2:
        raise ValueError(f'{label}: expected warmup + {sample_count} timed + accuracy requests; got {len(rows)}')
    diagnostics = [json.loads(line) for line in (directory / f'{label}.jsonl').read_text().splitlines()]
    if len(diagnostics) != len(rows) or any(row['status'] for row in diagnostics):
        raise ValueError('diagnostic request count or status mismatch')
    previous_bytes = 0
    for row, diagnostic in zip(rows, diagnostics):
        if row['io_bytes'] != diagnostic['read_bytes_total'] - previous_bytes:
            raise ValueError('NVTX I/O byte count does not match completed store reads')
        previous_bytes = diagnostic['read_bytes_total']
    timed = rows[1:1 + sample_count]
    totals = {key: sum(row[key] for row in timed) for key in
              ('total_ms', 'compute_ms', 'io_inflight_ms', 'overlap_ms',
               'same_expert_overlap_ms', 'io_without_compute_ms', 'neither_ms', 'io_bytes',
               'io_outstanding_time_ms')}
    summary = dict(label=label, timed_requests=sample_count, benchmark=benchmark,
                   mean_ms={key: value / sample_count for key, value in totals.items() if key.endswith('_ms')},
                   compute_overlap_percent=100 * totals['overlap_ms'] / totals['compute_ms'],
                   request_overlap_percent=100 * totals['overlap_ms'] / totals['total_ms'],
                   mean_outstanding_io=totals['io_outstanding_time_ms'] / totals['total_ms'],
                   read_gbytes_per_request_second=totals['io_bytes'] / totals['total_ms'] / 1e6,
                   first_request=rows[0], rows=rows,
                   compute_threads=sorted({event['thread'] for event in events if event['kind'] == 'COMPUTE'}),
                   io_threads=sorted({event['thread'] for event in events if event['kind'] == 'WEIGHT_IO'}))
    (directory / f'{label}_overlap.json').write_text(json.dumps(summary, indent=2))
    return {key: value for key, value in summary.items() if key not in ('rows', 'benchmark')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summaries = [analyze(args.directory, label) for label in ('expert', 'matrix')]
    (args.directory / 'overlap_summary.json').write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
