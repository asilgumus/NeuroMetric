"""Choose an equal-weight checkpoint ensemble using validation predictions only."""
from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def scores(y, p):
    error = np.asarray(p) - np.asarray(y)
    return {
        'mae': float(np.mean(np.abs(error))),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'r2': float(r2_score(y, p)),
        'mean_gap': float(np.mean(error)),
        'within_5_years': float(np.mean(np.abs(error) <= 5)),
        'within_10_years': float(np.mean(np.abs(error) <= 10)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True,
                        help='member=training-output-directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runs = {}
    for spec in args.run:
        member, folder = spec.split('=', 1)
        runs[int(member)] = Path(folder)
    if set(runs) != set(range(5)):
        raise ValueError('Exactly CNN1 members 0,1,2,3,4 are required')
    frames = {}
    reference = None
    for member, folder in sorted(runs.items()):
        frame = pd.read_csv(folder / 'predictions.csv').sort_values(['split', 'subject_id']).reset_index(drop=True)
        key = frame[['split', 'subject_id', 'age']]
        if reference is not None:
            pd.testing.assert_frame_equal(reference, key)
        reference = key
        frames[member] = frame
    val = reference.split == 'val'
    candidates = []
    # Search subsets only on validation. Prefer fewer members on exact ties.
    for size in range(1, 6):
        for members in combinations(range(5), size):
            prediction = np.mean([frames[m].loc[val, 'finetuned'].to_numpy() for m in members], axis=0)
            candidates.append({'members': members, **scores(reference.loc[val, 'age'], prediction)})
    candidates.sort(key=lambda x: (x['mae'], len(x['members']), x['members']))
    chosen = candidates[0]['members']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = reference.copy()
    for member in range(5):
        result[f'member_{member}'] = frames[member].finetuned
    result['ensemble'] = result[[f'member_{m}' for m in chosen]].mean(axis=1)
    result['brain_age_gap'] = result.ensemble - result.age
    metrics = {}
    for split in ['val', 'test']:
        mask = result.split == split
        metrics[split] = scores(result.loc[mask, 'age'], result.loc[mask, 'ensemble'])
    selected = []
    for member in chosen:
        source = runs[member] / 'best_model.pt'
        target = output / f'member_{member}.pt'
        shutil.copy2(source, target)
        selected.append({'member': member, 'file': target.name,
                         'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest = {'selection_data': 'validation_only', 'weighting': 'equal_mean',
                'selected_members': list(chosen), 'checkpoints': selected,
                'validation_candidates': [{**c, 'members': list(c['members'])} for c in candidates],
                'metrics': metrics}
    result.to_csv(output / 'predictions.csv', index=False)
    (output / 'ensemble.json').write_text(json.dumps(manifest, indent=2))
    test = result[result.split == 'test']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(test.age, test.ensemble, s=15, alpha=.7)
    limits = [test.age.min(), test.age.max()]
    axes[0].plot(limits, limits, 'k--')
    axes[0].set(xlabel='Chronological age', ylabel='Ensemble brain age')
    axes[1].scatter(test.age, test.brain_age_gap, s=15, alpha=.7)
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set(xlabel='Chronological age', ylabel='Brain age gap (years)')
    fig.tight_layout()
    fig.savefig(output / 'test_evaluation.png', dpi=160)
    plt.close(fig)
    report = f'''# BrainAGE v2 ensemble raporu

CNN1 üyeleri {', '.join(map(str, chosen))}, validation MAE üzerinden seçildi ve
eşit ağırlıkla ortalandı. Test verisi ensemble seçimine katılmadı.

- Validation MAE: {metrics['val']['mae']:.3f} yıl
- Test MAE: {metrics['test']['mae']:.3f} yıl
- Test RMSE: {metrics['test']['rmse']:.3f} yıl
- Test R²: {metrics['test']['r2']:.3f}
- ±5 yıl oranı: %{100*metrics['test']['within_5_years']:.1f}
- ±10 yıl oranı: %{100*metrics['test']['within_10_years']:.1f}
- Ortalama brain-age gap: {metrics['test']['mean_gap']:+.3f} yıl

Bu, bağımsız hastane doğrulaması olmayan bir araştırma prototipidir. Bölgesel
yaş, hastalık tanısı veya klinik öneri üretmez.
'''
    (output / 'REPORT_TR.md').write_text(report)
    print(json.dumps({'selected_members': list(chosen), 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
