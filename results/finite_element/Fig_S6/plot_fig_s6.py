from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ROOT = Path(__file__).resolve().parent
FAMILIES = [('R6', 'Random-6', '#2F5597'), ('SL', 'Single-loop', '#C55A11'),
            ('R12', 'Random-12', '#1E5A29'), ('DL', 'Double-loops', '#6A51A3')]
PANELS = [('load_path_curves.csv', 'axial_load_participation', 'Axial-load participation', (.15, .65)),
          ('load_path_curves.csv', 'internal_axial_load_fraction', 'Internal axial-load fraction', (.25, .85)),
          ('energy_curves.csv', 'energy_participation', 'Strain-energy participation', (.1, .6)),
          ('energy_curves.csv', 'internal_energy_fraction', 'Internal strain-energy fraction', (.15, .75))]

def main():
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11,
                         'font.weight': 'normal', 'axes.labelweight': 'normal',
                         'axes.linewidth': .85, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.7))
    grid = np.linspace(.01, .5, 100)
    individual, statistics = [], []
    for i, (ax, (file, metric, label, limits)) in enumerate(zip(axes.flat, PANELS)):
        raw = pd.read_csv(ROOT / 'data' / file)
        for family, name, color in FAMILIES:
            values = []
            groups = raw[raw.family == family].groupby('sample', sort=True)
            assert len(groups) == 20
            for sid, group in groups:
                group = group.sort_values('strain').drop_duplicates('strain')
                assert group.strain.min() <= grid[0] and group.strain.max() >= .499
                curve = np.interp(grid, group.strain, group[metric])
                values.append(curve)
                individual.extend({'panel': chr(97+i), 'sample': sid, 'family': family,
                                   'strain': x, 'metric': metric, 'value': y}
                                  for x, y in zip(grid, curve))
            values = np.asarray(values)
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.fill_between(grid, mean-sd, mean+sd, color=color, alpha=.14, lw=0)
            ax.plot(grid, mean, color=color, lw=1.65, label=name)
            statistics.extend({'panel': chr(97+i), 'family': family, 'strain': x,
                               'metric': metric, 'n': 20, 'mean': y, 'SD': s}
                              for x, y, s in zip(grid, mean, sd))
        ax.set(xlim=(0, .5), ylim=limits, xlabel='Engineering strain', ylabel=label)
        ax.xaxis.set_major_locator(MultipleLocator(.1))
        ax.yaxis.set_major_locator(MultipleLocator(.1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(direction='in', top=True, right=True, length=3.5, width=.85)
        ax.minorticks_off()
        ax.text(-.19, 1.045, '('+chr(97+i)+')', transform=ax.transAxes, fontsize=13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(.53, 1.005), handlelength=1.8, columnspacing=1.5)
    fig.subplots_adjust(left=.12, right=.98, bottom=.10, top=.895, wspace=.36, hspace=.37)
    target = ROOT / 'figures'
    target.mkdir(exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(target / ('Fig_S6_load_path_energy.'+ext), dpi=600, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(individual).to_csv(ROOT/'data'/'plotted_individual_curves.csv', index=False)
    pd.DataFrame(statistics).to_csv(ROOT/'data'/'plotted_mean_SD.csv', index=False)

if __name__ == '__main__':
    main()
