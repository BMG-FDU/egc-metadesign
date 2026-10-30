from pathlib import Path
import csv, json, hashlib, shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '.runtime'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
COLORS = ['#2F5597', '#C55A11', '#1E5A29', '#6A51A3', '#238A8D']
SAMPLES = ['P1', 'P2', 'P3', 'P4', 'P5']
VARIANTS = ['baseline', 'section0475', 'top_rotation_fixed', 'bottom_rotation_free', 'lateral_free']
VARIANT_LABELS = ['Baseline', 'Section\n'+r'$\nu_s=0.475$', 'Top rotation\nfixed', 'Bottom rotation\nfree', 'Lateral\nrelease']
BOUNDARY = VARIANTS[2:]

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def raw_curve(sample, variant):
    if variant == 'baseline':
        path = ROOT / 'data/raw_baseline' / f'{sample}_nominal_curve.csv'
        volume_col = 'nominal_beam_volume_mm3'
        kind = 'nominal beam, section nu=0'
    elif variant == 'section0475':
        path = ROOT / 'data/new0475' / f'{sample}_section0475_curve.csv'
        volume_col = 'nominal_beam_volume_mm3'
        kind = 'new beam, section nu=0.475'
    else:
        path = ROOT / 'data/raw_baseline' / f'{sample}_{variant}_curve.csv'
        volume_col = 'nominal_beam_volume_mm3'
        kind = variant
    rows = read_csv(path)
    strain = np.array([float(r['strain']) for r in rows])
    u = np.array([float(r['U2_mm']) for r in rows])
    force = np.array([float(r['RF2_N']) for r in rows])
    if len(strain) < 2 or np.any(np.diff(strain) < -1e-7) or np.any(np.diff(u) < -1e-9):
        raise ValueError(f'Invalid or nonmonotone RF-U curve: {path}')
    work = np.r_[0.0, np.cumsum(np.diff(u) * (force[1:] + force[:-1]) / 2)] / 1000.0
    model = next(r for r in MODELS if r['label'] == sample)
    volume = float(model[volume_col])
    mass = volume * 1.15e-6
    return dict(path=path, kind=kind, strain=strain, u=u, force=force,
                work=work, sea=work/mass, mass=mass, volume=volume)

def interp(curve, field, x):
    return np.interp(x, curve['strain'], curve[field])

def nrmse(a, b):
    span = float(np.ptp(b))
    return float(np.sqrt(np.mean((a-b)**2))/span*100) if span else float('nan')

def main():
    global MODELS
    cfg = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    if cfg.get('figure') != 'S5' or cfg.get('material_nu') != 0.475 or cfg.get('response_scale') != 1:
        raise ValueError('Config mismatch: expected S5, material nu=0.475, response_scale=1.')
    MODELS = read_csv(ROOT/'data/models.csv')
    # Recreate output from current inputs so no stale S4 or intermediate files survive.
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,
        'xtick.labelsize':7.1,'ytick.labelsize':7.1,'axes.linewidth':.7,
        'xtick.direction':'in','ytick.direction':'in','pdf.fonttype':42,'ps.fonttype':42,
        'mathtext.fontset':'stix','font.weight':'normal'})

    audit, plot_data, metrics = [], [], []
    terminal_data = {}
    for sample in SAMPLES[:3]:
        curves = {v: raw_curve(sample, v) for v in VARIANTS}
        # A single matched endpoint per sample across all three boundary conditions,
        # the corrected section curve, and the nominal reference.
        common_end = min(0.5, *(float(c['strain'][-1]) for c in curves.values()))
        grid = np.linspace(0.0, common_end, 501)
        base = curves['baseline']
        for variant, curve in curves.items():
            audit.append({'sample':sample,'variant':variant,'file':str(curve['path'].relative_to(ROOT)),
                'model_description':curve['kind'],'section_nu':0 if variant in ('baseline',*BOUNDARY) else 0.475,
                'material_nu':0.475,'volume_mm3':curve['volume'],'mass_kg':curve['mass'],
                'source_end_strain':curve['strain'][-1],'common_comparison_end_strain':common_end,
                'integrated_work_at_common_end_J':interp(curve,'work',np.array([common_end]))[0],
                'SEA_at_common_end_J_kg':interp(curve,'sea',np.array([common_end]))[0]})
            for e, force, work, sea in zip(grid, interp(curve,'force',grid), interp(curve,'work',grid), interp(curve,'sea',grid)):
                plot_data.append({'sample':sample,'variant':variant,'strain':e,'force_N':force,
                                  'loading_work_J':work,'specific_tensile_loading_work_J_kg':sea})
        bforce, bwork, bsea = interp(base,'force',grid), interp(base,'work',grid), interp(base,'sea',grid)
        for variant, curve in curves.items():
            if variant == 'baseline':
                continue
            f,w,s = interp(curve,'force',grid), interp(curve,'work',grid), interp(curve,'sea',grid)
            metrics.append({'sample':sample,'comparison':f'{variant}_vs_baseline','common_end_strain':common_end,
                'reference_work_J':bwork[-1],'variant_work_J':w[-1],'work_change_pct':(w[-1]/bwork[-1]-1)*100,
                'reference_SEA_J_kg':bsea[-1],'variant_SEA_J_kg':s[-1],'SEA_change_pct':(s[-1]/bsea[-1]-1)*100,
                'force_curve_NRMSE_pct':nrmse(f,bforce),'loading_work_curve_NRMSE_pct':nrmse(w,bwork),
                'SEA_curve_NRMSE_pct':nrmse(s,bsea)})
        terminal_data[sample] = curves

    write_csv('FigS5a_actual_plot_data.csv', plot_data)
    write_csv('FigS5a_full_statistics.csv', metrics)
    write_csv('raw_curve_audit.csv', audit)

    # Original two-panel layout and physical quantities retained.
    fig, axs = plt.subplots(1,2,figsize=(7.35,3.0))
    fig.subplots_adjust(wspace=.37,top=.79,bottom=.23)
    x = np.arange(5)
    for i, sample in enumerate(['P1','P3','P2']):
        curves = terminal_data[sample]
        for vi, variant in enumerate(VARIANTS):
            c=curves[variant]
            sea_end=float(np.interp(min(.5,c['strain'][-1]),c['strain'],c['sea']))
            # For terminal bars, all variants use the same within-sample matched end.
            sea_end=float(np.interp(min(.5,*(float(terminal_data[sample][v]['strain'][-1]) for v in VARIANTS)),c['strain'],c['sea']))
            axs[0].plot(vi,sea_end,marker='o',ms=3.5,color=COLORS[i],lw=0)
            terminal_data[sample][variant]['plot_terminal_sea']=sea_end
        axs[0].plot(x,[curves[v]['plot_terminal_sea'] for v in VARIANTS],'-o',color=COLORS[i],lw=1.1,ms=3.5,label=sample)
    axs[0].set_ylabel('Specific tensile loading work (J/kg)')
    axs[0].set_xticks(x,VARIANT_LABELS,rotation=0)
    axs[0].tick_params(axis='x',labelsize=6.4,pad=3)
    axs[0].legend(frameon=False,fontsize=7.1)

    local = read_csv(ROOT/'data/original_panels/FigS5b_local_strain.csv')
    write_csv('FigS5b_local_strain_plot_data.csv',local)
    for variant,col,offset,label in [('nominal',COLORS[0],-.1,'Beam'),('contact',COLORS[1],.1,'Solid')]:
        rr=[next(r for r in local if r['sample']==p and r['variant']==variant) for p in SAMPLES]
        xx=np.arange(5)+offset
        axs[1].scatter(xx,[float(r['peak_engineering_strain']) for r in rr],s=16,color=col,label=label+' maximum')
        axs[1].scatter(xx,[float(r['final_p99_engineering_strain']) for r in rr],s=16,facecolors='none',edgecolors=col,label=label+' 99th percentile')
    axs[1].axhline(1.522504,color='#9E9E9E',ls='--',lw=1)
    axs[1].set_xticks(np.arange(5),SAMPLES)
    axs[1].set_ylabel('Local engineering strain')
    axs[1].legend(frameon=False,fontsize=7.1,ncol=2,loc='lower left',bbox_to_anchor=(0,1.01))
    for i,ax in enumerate(axs):
        ax.text(-.13,1.035,f'({chr(97+i)})',transform=ax.transAxes,fontsize=8.5,va='bottom',fontweight='normal')
        ax.grid(False); ax.minorticks_off(); ax.tick_params(direction='in',length=2.4,width=.65)
        if i==1:
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.arange(5)))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'FigS5_dynamic_sensitivity_local_strain.{ext}',dpi=600,bbox_inches='tight')
    plt.close(fig)

    # Verify copied file hashes against the handoff manifest.
    provenance=json.loads((ROOT/'文件来源.json').read_text(encoding='utf-8'))
    hashes=[]
    for item in provenance:
        p=ROOT/item['local']
        actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ''
        hashes.append({'local':item['local'],'exists':p.exists(),'expected_sha256':item['sha256'],
                       'actual_sha256':actual,'match':actual==item['sha256']})
    write_csv('input_hash_audit.csv',hashes)
    local_max=max(float(r['peak_engineering_strain']) for r in local)
    analysis = ['# Fig. S5 统一复算与重绘结果','',
        '所有梁曲线均从原始 RF2–U2 数据按梯形法重新积分；材料泊松比为 0.475，基准梁截面泊松比为 0，新敏感性梁截面泊松比为 0.475。质量按模型体积乘 1.15e-6 kg/mm³。',
        '边界条件三条曲线分别与同一 P 样本基准曲线比较，且采用该样本所有绘制曲线共同覆盖的应变终点。力、加载功、单位质量加载功曲线 NRMSE 分别计算。',
        'S5(a) 表示 specific tensile loading work（单位质量拉伸加载功），单位为 J/kg。S5(b) 使用原名义梁和接触实体局部应变数据。',
        f'输入来源清单共 {len(hashes)} 项，SHA256 匹配 {sum(bool(r["match"]) for r in hashes)} 项；来源记录见 input_hash_audit.csv。',
        f'局部最大应变数据的最高峰值为 {local_max:.4f}；局部应变面板保留原始图示数据。','',
        '## 结论数据（共同终点）','',
        '|试样|截面0.475相对基准SEA变化|顶部转动固定|底部转动自由|侧向释放|共同终点应变|',
        '|---|---:|---:|---:|---:|---:|']
    for sample in ['P1','P2','P3']:
        by={r['comparison']:r for r in metrics if r['sample']==sample}
        end=metrics[[r['sample'] for r in metrics].index(sample)]['common_end_strain']
        analysis.append(f"|{sample}|{by['section0475_vs_baseline']['SEA_change_pct']:+.2f}%|{by['top_rotation_fixed_vs_baseline']['SEA_change_pct']:+.2f}%|{by['bottom_rotation_free_vs_baseline']['SEA_change_pct']:+.2f}%|{by['lateral_free_vs_baseline']['SEA_change_pct']:+.2f}%|{end:.6f}|")
    analysis += ['', '## 英文图注','',
        'Figure S5. Sensitivity of the tensile response to beam-section Poisson’s ratio and boundary conditions. (a) Specific tensile loading work (J/kg), recalculated by trapezoidal integration of the original reaction-force–displacement curves and evaluated at the common attained strain for each specimen. The material Poisson’s ratio was 0.475; the baseline and sensitivity beam-section Poisson’s ratios were 0 and 0.475, respectively. Boundary-condition cases are compared with the baseline over the same attained strain interval. (b) Peak and final 99th-percentile local engineering strains in the original nominal beam and contacting solid models.']
    (OUT/'结果分析.md').write_text('\n'.join(analysis)+'\n',encoding='utf-8')
    print(f'Created {OUT} with {len(metrics)} matched comparisons; provenance SHA matches {sum(bool(r["match"]) for r in hashes)}/{len(hashes)}.')

if __name__ == '__main__':
    main()
