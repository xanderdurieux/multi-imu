"""Plots for synchronisation method comparison and selected alignment quality."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_dataframe, recording_stage_dir

from .common import add_vector_norms, remove_dropouts
from .helpers import METHOD_LABELS, METHOD_ORDER, METHOD_STAGES


def _load_sync_summary(recording_name: str) -> dict:
    path = recording_stage_dir(recording_name, 'synced') / 'all_methods.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _load_sensor(recording_name: str, stage: str, sensor: str) -> pd.DataFrame:
    path = recording_stage_dir(recording_name, stage) / f'{sensor}.csv'
    if not path.exists():
        return pd.DataFrame()
    return remove_dropouts(add_vector_norms(load_dataframe(path)))


def plot_method_comparison(recording_name: str) -> Path | None:
    payload = _load_sync_summary(recording_name)
    if not payload:
        return None

    methods_payload = payload.get('methods', {})
    selected_method = payload.get('selected_method')
    methods = [method for method in METHOD_ORDER if methods_payload.get(method, {}).get('available')]
    if not methods:
        return None

    corr = [methods_payload[method].get('corr_offset_and_drift') for method in methods]
    drift = [abs(methods_payload[method].get('drift_ppm') or 0.0) for method in methods]
    labels = [METHOD_LABELS[method] for method in methods]
    colors = ['#d62728' if method == selected_method else '#4c9be8' for method in methods]
    y = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(11, max(4, len(methods) * 0.8 + 1.5)), constrained_layout=True)
    axes[0].barh(y, corr, color=colors)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel('acc_norm correlation')
    axes[0].set_title('Alignment quality')
    axes[0].grid(True, axis='x', alpha=0.3)

    axes[1].barh(y, drift, color=colors)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel('drift magnitude [ppm]')
    axes[1].set_title('Estimated drift')
    axes[1].grid(True, axis='x', alpha=0.3)

    out_path = recording_stage_dir(recording_name, 'synced') / 'sync_method_comparison.png'
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_alignment(recording_name: str, *, selected_stage: str | None = None) -> Path | None:
    selected_stage = selected_stage or _load_sync_summary(recording_name).get('selected_stage', 'synced')
    parsed_sporsa = _load_sensor(recording_name, 'parsed', 'sporsa')
    parsed_arduino = _load_sensor(recording_name, 'parsed', 'arduino')
    synced_sporsa = _load_sensor(recording_name, selected_stage, 'sporsa')
    synced_arduino = _load_sensor(recording_name, selected_stage, 'arduino')
    if any(df.empty for df in (parsed_sporsa, parsed_arduino, synced_sporsa, synced_arduino)):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    panels = [
        ('Before sync', parsed_sporsa, parsed_arduino, False),
        ('After sync', synced_sporsa, synced_arduino, True),
    ]
    for ax, (title, sporsa_df, arduino_df, shared_clock) in zip(axes, panels):
        for label, df, color in (('sporsa', sporsa_df, '#e05c44'), ('arduino', arduino_df, '#4c9be8')):
            ts = df['timestamp'].to_numpy(dtype=float)
            base = min(sporsa_df['timestamp'].iloc[0], arduino_df['timestamp'].iloc[0]) if shared_clock else ts[0]
            time_s = (ts - float(base)) / 1000.0
            ax.plot(time_s, df['acc_norm'].to_numpy(dtype=float), lw=0.7, alpha=0.8, label=label, color=color)
        ax.set_title(title)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('acc_norm')
        ax.grid(True, alpha=0.3)
        ax.legend()

    out_path = recording_stage_dir(recording_name, 'synced') / 'sync_alignment.png'
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out_path


def generate_sync_plots(recording_name: str) -> list[Path]:
    outputs = []
    for path in (plot_method_comparison(recording_name), plot_alignment(recording_name)):
        if path is not None:
            outputs.append(path)
    return outputs
