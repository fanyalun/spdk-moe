#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics

from analyze import duration, intersection, merge


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(directory):
    with sqlite3.connect(directory / 'lattice.sqlite') as connection:
        raw = connection.execute('SELECT start, end, text, globalTid FROM NVTX_EVENTS ORDER BY start').fetchall()
    pattern = re.compile(r'^(REQUEST|COMPUTE|DECODE|WEIGHT_IO|ROUTE|COMBINE) '
                         r'expert=(-?\d+) stage=(-?\d+) bytes=(\d+)$')
    events = []
    for start, end, text, thread in raw:
        match = pattern.match((text or '').split(' | meta:')[-1])
        if match:
            if end is None or end <= start:
                raise ValueError(f'incomplete NVTX interval: {text}')
            kind, expert, stage, size = match.groups()
            events.append(dict(start=start, end=end, kind=kind, expert=int(expert),
                               stage=int(stage), size=int(size), thread=thread))
    requests = [event for event in events if event['kind'] == 'REQUEST']
    diagnostics = load_rows(directory / 'lattice.jsonl')
    clients = load_rows(directory / 'lattice_requests.jsonl')
    manifest = json.loads((directory / 'manifest.json').read_text())
    warmup = manifest['warmup_count']
    windowed = len(requests) == len(clients)
    if len(clients) != manifest['count'] or (not windowed and len(requests) != warmup + len(clients) + 1):
        raise ValueError('NVTX request count differs from warmup + timed + accuracy')
    if len(diagnostics) != warmup + len(clients) + 1 or any(row['status'] for row in diagnostics):
        raise ValueError('diagnostic request count/status mismatch')
    rows = []
    previous_bytes = previous_decoded = previous_misses = 0
    if windowed:
        prior = diagnostics[warmup - 1]
        previous_bytes = prior['read_bytes_total']
        previous_decoded = prior['decoded_bytes_total']
        previous_misses = prior['cache_misses_total']
        diagnostics = diagnostics[warmup:warmup + len(clients)]
        warmup_offset = 0
    else:
        warmup_offset = warmup
    for index, (request, diagnostic) in enumerate(zip(requests, diagnostics)):
        contained = [event for event in events if event['kind'] != 'REQUEST' and
                     request['start'] <= event['start'] and event['end'] <= request['end']]
        by_kind = {kind: [event for event in contained if event['kind'] == kind]
                   for kind in ('COMPUTE', 'DECODE', 'WEIGHT_IO', 'ROUTE', 'COMBINE')}
        intervals = {kind: merge((event['start'], event['end']) for event in group)
                     for kind, group in by_kind.items()}
        compute, decode, io = (intervals[kind] for kind in ('COMPUTE', 'DECODE', 'WEIGHT_IO'))
        work = merge(compute + decode)
        work_io = duration(intersection(work, io))
        compute_io = duration(intersection(compute, io))
        decode_io = duration(intersection(decode, io))
        compute_decode = duration(intersection(compute, decode))
        io_bytes = sum(event['size'] for event in by_kind['WEIGHT_IO'])
        decoded_bytes = sum(event['size'] for event in by_kind['DECODE'])
        misses = diagnostic['cache_misses_total'] - previous_misses
        if io_bytes != diagnostic['read_bytes_total'] - previous_bytes:
            raise ValueError('NVTX I/O bytes differ from completed store bytes')
        if decoded_bytes != diagnostic['decoded_bytes_total'] - previous_decoded:
            raise ValueError('NVTX decode bytes differ from restored store bytes')
        if len(by_kind['DECODE']) != len(by_kind['WEIGHT_IO']):
            raise ValueError('each completed read must have one decoder range')
        previous_bytes = diagnostic['read_bytes_total']
        previous_decoded = diagnostic['decoded_bytes_total']
        previous_misses = diagnostic['cache_misses_total']
        row = dict(request=index, total_ms=(request['end'] - request['start']) / 1e6,
            compute_ms=duration(compute) / 1e6, decode_ms=duration(decode) / 1e6,
            io_inflight_ms=duration(io) / 1e6, work_union_ms=duration(work) / 1e6,
            compute_io_overlap_ms=compute_io / 1e6, decode_io_overlap_ms=decode_io / 1e6,
            compute_decode_overlap_ms=compute_decode / 1e6, work_io_overlap_ms=work_io / 1e6,
            work_outside_io_ms=(duration(work) - work_io) / 1e6,
            io_without_work_ms=(duration(io) - work_io) / 1e6,
            neither_ms=((request['end'] - request['start']) - duration(merge(work + io))) / 1e6,
            route_ms=duration(intervals['ROUTE']) / 1e6,
            combine_ms=duration(intervals['COMBINE']) / 1e6,
            io_bytes=io_bytes, decoded_bytes=decoded_bytes, misses=misses,
            io_ranges=len(by_kind['WEIGHT_IO']), decode_ranges=len(by_kind['DECODE']),
            io_outstanding_ms=sum(e['end'] - e['start'] for e in by_kind['WEIGHT_IO']) / 1e6)
        if warmup_offset <= index < warmup_offset + len(clients):
            row['timed_request'] = index - warmup_offset
            row['tcp_latency_ms'] = clients[index - warmup_offset]['latency_ns'] / 1e6
        rows.append(row)
    timed = rows[warmup_offset:warmup_offset + len(clients)]
    numeric = [key for key, value in timed[0].items()
               if isinstance(value, (int, float)) and key not in ('request', 'timed_request')]
    means = {key: statistics.mean(row[key] for row in timed) for key in numeric}

    def percent(numerator, denominator):
        return 100 * sum(row[numerator] for row in timed) / sum(row[denominator] for row in timed)

    summary = dict(timed_requests=len(timed), warmup_requests=warmup,
        input_digest=hashlib.sha256(''.join(row['input_hash'] for row in clients).encode()).hexdigest(),
        mean=means, tcp_p99_ms=max(row['tcp_latency_ms'] for row in timed),
        target_p99_ms=max(row['total_ms'] for row in timed),
        compute_covered_by_io_percent=percent('compute_io_overlap_ms', 'compute_ms'),
        decode_covered_by_io_percent=percent('decode_io_overlap_ms', 'decode_ms'),
        combined_work_covered_by_io_percent=percent('work_io_overlap_ms', 'work_union_ms'),
        stream_decode_gbps=sum(row['decoded_bytes'] for row in timed) /
            sum(row['decode_ms'] for row in timed) / 1e6,
        decode_ms_per_miss=sum(row['decode_ms'] for row in timed) / sum(row['misses'] for row in timed),
        mean_io_outstanding=means['io_outstanding_ms'] / means['total_ms'],
        threads={kind: sorted({event['thread'] for event in events if event['kind'] == kind})
                 for kind in ('COMPUTE', 'DECODE', 'WEIGHT_IO')},
        rows=rows, notes='Instrumented AIO; software observed outstanding I/O, not PCIe wire time. '
            'NVTX capture start/stop may delay boundary TCP requests; not P99 qualification.')
    (directory / 'lattice_overlap.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}, indent=2))
    plot(directory, events, timed)


def plot(directory, events, timed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    representatives = [('Median request', min(timed, key=lambda row:
        abs(row['tcp_latency_ms'] - statistics.median(r['tcp_latency_ms'] for r in timed)))),
        ('P99 request', max(timed, key=lambda row: row['tcp_latency_ms']))]
    requests = [event for event in events if event['kind'] == 'REQUEST']
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), constrained_layout=True)
    tracks = [('WEIGHT_IO', 'I/O outstanding', '#258ab0'),
              ('DECODE', 'Decode on owner', '#e49331'),
              ('COMPUTE', 'Expert compute', '#37965d'),
              ('ROUTE', 'Route / Top-K', '#8a67aa'), ('COMBINE', 'Combine', '#8a67aa')]
    for ax, (title, row) in zip(axes, representatives):
        request = requests[row['request']]
        for track, (kind, label, color) in enumerate(tracks):
            intervals = merge((event['start'], event['end']) for event in events
                if event['kind'] == kind and request['start'] <= event['start'] and
                event['end'] <= request['end'])
            polygons = [[((start - request['start']) / 1e6, track - 0.3),
                         ((end - request['start']) / 1e6, track - 0.3),
                         ((end - request['start']) / 1e6, track + 0.3),
                         ((start - request['start']) / 1e6, track + 0.3)]
                        for start, end in intervals]
            ax.add_collection(PolyCollection(polygons, facecolors=color, edgecolors='none'))
        ax.set_yticks(range(len(tracks)), [label for _, label, _ in tracks])
        ax.set_ylim(len(tracks) - 0.4, -0.6)
        ax.set_xlim(0, row['total_ms'])
        ax.set_xlabel('Time from target request start (ms)')
        ax.set_title(f'{title} #{row["timed_request"]}: target {row["total_ms"]:.2f} ms; '
                     f'decode {row["decode_ms"]:.2f} ms; compute {row["compute_ms"]:.2f} ms')
        ax.grid(axis='x', alpha=0.2)
    fig.suptitle('Reversible MoE profile: union of NVTX intervals; local AIO simulation')
    fig.savefig(directory / 'lattice_timeline.png', dpi=160)
    fig.savefig(directory / 'lattice_timeline.svg')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    analyze(parser.parse_args().directory)
