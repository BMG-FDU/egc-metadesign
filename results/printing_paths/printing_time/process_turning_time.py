from pathlib import Path
import csv
import statistics
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'output')
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
source = Path(__file__).resolve().parent / 'data' / 'turning_time_replicates_raw.csv'
rows = []
with source.open(encoding='utf-8-sig', newline='') as f:
    for fields in csv.reader(f):
        if not fields:
            continue
        angle = int(fields[0])
        times = [float(v) for v in fields[1:4]]
        mean = statistics.mean(times)
        assert abs(mean - float(fields[4])) < 1e-5
        rows.append([angle, *times, mean, statistics.stdev(times), len(times)])
assert len(rows) == 18 and sum(r[-1] for r in rows) == 54
with (out / 'turning_time_statistics.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['angle_deg', 'trial_1_s', 'trial_2_s', 'trial_3_s', 'mean_s', 'sample_SD_s', 'n'])
    writer.writerows(rows)

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
                     'axes.labelsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'font.weight': 'normal', 'axes.labelweight': 'normal',
                     'axes.linewidth': 0.8, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(4.4, 3.4))
blue, orange = '#2F5597', '#C55A11'
for k, offset in enumerate([-1, 0, 1]):
    ax.scatter([r[0] + offset for r in rows], [r[1+k] for r in rows],
               s=9, facecolor='white', edgecolor=orange, linewidth=0.65, alpha=0.9,
               label='Individual prints' if k == 0 else None, zorder=3)
ax.errorbar([r[0] for r in rows], [r[4] for r in rows], yerr=[r[5] for r in rows],
            fmt='o-', color=blue, markersize=2.6, linewidth=1.2,
            elinewidth=0.8, capsize=2.2, capthick=0.8, label='Mean ± SD', zorder=4)
ax.axvline(150, color='#4D4D4D', linestyle='--', linewidth=0.8, label='150° criterion')
ax.set(xlabel='Turning angle (°)', ylabel='Printing time (s)',
       xlim=(-5, 180), ylim=(60, 170), xticks=[0, 40, 80, 120, 160],
       yticks=[60, 80, 100, 120, 140, 160])
ax.tick_params(direction='in', length=3.5, width=0.8, top=False, right=False)
ax.minorticks_off()
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(t) for t in ['Mean ± SD', 'Individual prints', '150° criterion']]
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          frameon=False, fontsize=10, loc='upper left', handlelength=1.6,
          labelspacing=0.35, borderpad=0.25)
fig.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.97)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(out / ('turning_time_replicates.' + ext), dpi=600)
plt.close(fig)

print(f'Saved {len(rows)} angles and {sum(r[-1] for r in rows)} measurements to {out}')
