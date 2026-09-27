"""All final scientific figures and a standalone HTML/PDF, sourced from results."""
import base64
import csv
import html
import json
import os
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from astropy.io import fits
from core import ROOT, calibrate, sha256
from experiment import load_records, metrics, write_csv

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':130,'savefig.dpi':160,'axes.labelcolor':'#263645',
                     'axes.titleweight':'bold','axes.grid':False})
BLUE='#23638c'; RED='#bd4b3e'


def read_csv(name):return list(csv.DictReader((ROOT/'results'/name).open()))


def save(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.png',bbox_inches='tight')
    plt.close(fig)


def build_figures(records,s,f,v):
    saved=np.load(ROOT/'cache/selected_masks.npz');keep=saved['keep'];score=saved['score'];e=saved['exposure'];roi=saved['roi'];size=f['tile_size']
    test=np.array([r['meta']['split']=='test' for r in records])
    rows=read_csv('operating_points.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained',sharey=True)
    methods=['isolation_forest','combined_robust','dynamic_robust','persistent_sideband','low_only','high_only','noise_only']
    colors=plt.cm.tab10(np.arange(7))
    for ax,split in zip(axes,['validation','test']):
        for color,m in zip(colors,methods):
            for sz in [32,64]:
                rr=sorted([r for r in rows if r['method']==m and r['split']==split and int(r['tile_size'])==sz],key=lambda r:float(r['exposure_fraction']))
                ax.plot([float(r['exposure_fraction']) for r in rr],[float(r['background_ratio']) for r in rr],
                    color=color,lw=1.2,alpha=.8,linestyle='-' if sz==32 else ':',label=m.replace('_',' ') if sz==32 else None)
        rr=sorted([r for r in rows if r['split']==split and r['pareto']=='True'],key=lambda r:float(r['exposure_fraction']))
        ax.plot([float(r['exposure_fraction']) for r in rr],[float(r['background_ratio']) for r in rr],color='black',lw=1.3,ls='--',label='empirical Pareto frontier')
        p=f if split=='validation' else s
        ax.scatter(p['exposure_fraction'],p['background_ratio'],marker='*',s=170,color='black',zorder=10,label='validation-selected point')
        if split=='test':
            ci=s['bootstrap_95_intervals']; x=p['exposure_fraction'];y=p['background_ratio']
            ax.errorbar(x,y,xerr=[[x-ci['exposure_fraction'][0]],[ci['exposure_fraction'][1]-x]],yerr=[[y-ci['background_ratio'][0]],[ci['background_ratio'][1]-y]],color='black',capsize=3)
        ax.axvline(.8,color='gray',ls=':',lw=1);ax.axhline(1,color='gray',lw=.7)
        ax.set(xlabel='Retained effective exposure fraction',ylabel='ROI background-rate ratio',title=split.capitalize(),xlim=(.52,1.01),ylim=(0,1.08))
        ax.grid(alpha=.15)
    axes[0].legend(loc='upper left',fontsize=8,ncol=1)
    fig.suptitle('Background rate versus exposure: lower and farther right is better\nsolid: 32-pixel tiles; dotted: 64-pixel tiles; test curve is descriptive, not retuned',fontsize=12)
    save(fig,'tradeoff')

    bins=np.arange(.5,21.5,1);centers=(bins[1:]+bins[:-1])/2
    before=np.zeros(len(centers),int);after=before.copy()
    pixels=np.zeros(440,int)
    for i in np.flatnonzero(test):
        r=records[i];ok=keep[i,r['y']//size,r['x']//size]
        before+=np.histogram(r['charge'],bins)[0];after+=np.histogram(r['charge'][ok],bins)[0];pixels+=r['pixel_hist']
    spectrum=[dict(charge_bin_center=float(x),baseline_count=int(a),cleaned_count=int(b),baseline_rate_per_million_pixel_days=float(a/s['baseline_exposure']*1e6),cleaned_rate_per_million_pixel_days=float(b/s['retained_exposure']*1e6)) for x,a,b in zip(centers,before,after)]
    write_csv(ROOT/'results/spectrum.csv',spectrum)
    fig,ax=plt.subplots(figsize=(10,4.4),layout='constrained')
    for counts,exp,label,color in [(before,s['baseline_exposure'],'Geometric baseline',BLUE),(after,s['retained_exposure'],'Selected mask',RED)]:
        ax.errorbar(centers,counts/exp*1e6,yerr=np.sqrt(counts)/exp*1e6,drawstyle='steps-mid',marker='.',label=label,color=color,capsize=2)
    ax.axvspan(3.5,7.5,color='gold',alpha=.18,label='ROI [3.5, 7.5) e-')
    ax.set(yscale='log',xlabel='Reconstructed cluster charge [e-]',ylabel='Events / (million pixel-days) / 1e bin',title='Held-out charge spectra, normalized by each selection’s exposure')
    ax.legend();save(fig,'spectra')

    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    xx=np.linspace(-1,10,441);axes[0].stairs(pixels,xx,color=BLUE)
    axes[0].set(yscale='log',xlim=(-.7,3),xlabel='Calibrated pixel charge [e-]',ylabel='Pixel count',title='Electron peaks: held-out pixels')
    for group in sorted({(r['meta']['module'],r['meta']['hdu']) for r in records}):
        rr=[r['meta'] for r in records if (r['meta']['module'],r['meta']['hdu'])==group]
        axes[1].plot([r['image_id'] for r in rr],[r['gain_adu_per_e'] for r in rr],lw=1,label=f'{group[0]} / {group[1]}')
    axes[1].axvline(86.5,c='gray',ls=':');axes[1].axvline(108.5,c='gray',ls=':')
    axes[1].set(xlabel='Acquisition index',ylabel='Gain [ADU/e-]',title='Calibration by module / HDU');axes[1].legend(fontsize=7,ncol=2)
    save(fig,'calibration')

    representative=next(i for i,r in enumerate(records) if r['meta']['module']=='S101' and r['meta']['image_id']==109 and r['meta']['hdu']==0)
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    filename=next(r['filename'] for r in manifest if r['module']=='S101' and r['image_id']==109)
    data=Path(os.environ.get('SKIPPER_DATA_DIR','../pp_proc'))
    with fits.open(data/filename,memmap=False) as hdul:
        raw=hdul[0].data[:512,8:3080];q,_=calibrate(hdul[0].data)
        fig,axes=plt.subplots(4,1,figsize=(12,10),layout='constrained')
        for ax,img,title,cmap,lim in [(axes[0],raw,'Input processed ADU (central 99% display range)','viridis',np.percentile(raw,[.5,99.5])),
            (axes[1],q,'Calibrated charge, display clipped to [-0.5, 8] e-','viridis',[-.5,8]),
            (axes[2],score[representative],'Isolation Forest patch anomaly score (ROI excluded)','magma',[.3,.75]),
            (axes[3],1-keep[representative].astype(int),'Rejected 32x32 tiles (black = rejected); orange points = surviving ROI','binary',[0,1])]:
            im=ax.imshow(img,origin='lower',aspect='auto',extent=[8,3080,0,512],cmap=cmap,vmin=lim[0],vmax=lim[1]);fig.colorbar(im,ax=ax,shrink=.75)
            ax.set(title=title,ylabel='Stored row',xlabel='Stored column')
        r=records[representative];sel=(r['charge']>=3.5)&(r['charge']<7.5);ok=keep[representative,r['y']//size,r['x']//size]
        axes[3].scatter(r['x'][sel&ok]+8,r['y'][sel&ok],s=12,c='#ee7c16',linewidths=.3,edgecolors='white')
        fig.suptitle('Representative held-out image: S101, acquisition 109, HDU0',fontsize=13)
        save(fig,'representative')

    fig,axes=plt.subplots(8,2,figsize=(14,16),layout='constrained')
    group_rows=[]
    for j,group in enumerate(sorted({(r['meta']['module'],r['meta']['hdu']) for r in records})):
        ix=np.array([test[i] and (r['meta']['module'],r['meta']['hdu'])==group for i,r in enumerate(records)])
        a=roi[ix].sum(0);b=(roi*keep)[ix].sum(0)
        # Count maps use a shared scale within each group; exposure metrics are in the table.
        vmax=max(float(np.percentile(a,99.5)),1)
        for ax,img,label in [(axes[j,0],a,'before'),(axes[j,1],b,'after')]:
            im=ax.imshow(img,origin='lower',aspect='auto',extent=[8,3080,0,512],cmap='inferno',vmin=0,vmax=vmax)
            ax.set(title=f'{group[0]} / HDU{group[1]} — {label}',ylabel='Row');fig.colorbar(im,ax=ax,label='ROI count',shrink=.8)
        group_rows.append(dict(module=group[0],hdu=group[1],**metrics(e[ix].sum(),(e*keep)[ix].sum(),roi[ix].sum(),(roi*keep)[ix].sum())))
    for ax in axes[-1]:ax.set_xlabel('Stored column')
    fig.suptitle('Held-out spatial ROI maps: identical color scale within each before/after pair',fontsize=14)
    save(fig,'spatial');write_csv(ROOT/'results/per_group.csv',group_rows)

    temporal=[]
    for image in range(22,131):
        ix=np.array([r['meta']['image_id']==image for r in records])
        temporal.append(dict(image_id=image,**metrics(e[ix].sum(),(e*keep)[ix].sum(),roi[ix].sum(),(roi*keep)[ix].sum())))
    write_csv(ROOT/'results/temporal.csv',temporal)
    fig,axes=plt.subplots(2,1,figsize=(12,6),layout='constrained',sharex=True)
    ids=[r['image_id'] for r in temporal]
    axes[0].plot(ids,[r['baseline_roi_background_rate']*1e6 for r in temporal],label='Before',color=BLUE)
    axes[0].plot(ids,[r['final_roi_background_rate']*1e6 for r in temporal],label='After',color=RED)
    axes[0].set(ylabel='ROI events / million pixel-days',title='Acquisition-level behavior (all modules/HDUs together)');axes[0].legend()
    axes[1].plot(ids,[r['exposure_fraction'] for r in temporal],color=BLUE);axes[1].set(xlabel='Acquisition index',ylabel='Retained exposure fraction')
    for ax in axes:
        ax.axvspan(22,86.5,alpha=.07,color=BLUE);ax.axvspan(86.5,108.5,alpha=.08,color='orange');ax.axvspan(108.5,130,alpha=.06,color='green')
        ax.axvline(86.5,color='gray',ls=':');ax.axvline(108.5,color='gray',ls=':')
    axes[0].text(35,axes[0].get_ylim()[1]*.95,'Train');axes[0].text(90,axes[0].get_ylim()[1]*.95,'Validation');axes[0].text(115,axes[0].get_ylim()[1]*.95,'Test')
    save(fig,'temporal')

    abl=read_csv('ablations.csv');names=list(dict.fromkeys(r['ablation'] for r in abl))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,key,title in zip(axes,['exposure_fraction','background_ratio'],['Retained exposure','ROI rate ratio']):
        for d,split,color in [(-.18,'validation',BLUE),(.18,'test',RED)]:
            ax.bar(np.arange(len(names))+d,[float(next(r[key] for r in abl if r['ablation']==name and r['split']==split)) for name in names],width=.34,label=split,color=color)
        ax.set(xticks=np.arange(len(names)),xticklabels=[n.replace('_','\n') for n in names],title=title);ax.legend()
    fig.suptitle('Forest ablations: each threshold fixed at the same 15% training-exposure budget')
    save(fig,'ablations')
    return group_rows,rows


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


CAPTIONS={
 'tradeoff':'Every tested operating point; frontier includes comparison methods. The star is fixed by validation. Error bars: paired four-acquisition block bootstrap, 95% intervals, conditional on the model. Lower rate and greater exposure are preferred.',
 'spectra':'Clusters are reconstructed before masking; selection uses the anchor tile. Each spectrum is divided by its own surviving pixel-days. Illustrative Poisson errors do not include calibration or temporal systematics.',
 'calibration':'Electron peaks and gain stability. Vertical separators are the frozen acquisition boundaries. The baseline is active geometry only, without MOSKITA production masks.',
 'representative':'First test acquisition, chosen by index, not by best visual performance. Coordinates are stored FITS coordinates. The score has no direct ROI component input. Display clipping does not alter numerical calculations.',
 'spatial':'All eight module/HDU combinations. These are counts per 32x32 tile, not exposure-corrected rates. Each pair shares a scale clipped at the baseline 99.5th percentile for visibility; exposure-normalized results appear in the adjacent table.',
 'temporal':'No random pixel split: paired modules and four HDUs each stay in one acquisition block. Training performance is in-sample; only the final green block is held out.',
 'ablations':'Removing one component from the selected Forest descriptor, with refitting on training only. These checks were not used to retune the final method; lower rate at lower exposure is not automatically better.'}


def main():
    s=json.loads((ROOT/'results/summary.json').read_text());f=json.loads((ROOT/'results/selection.json').read_text());v=json.loads((ROOT/'results/verification.json').read_text())
    records=load_records(True);groups,operating=build_figures(records,s,f,v)
    baseline=json.loads((ROOT/'results/baseline_all.json').read_text()); ci=s['bootstrap_95_intervals']
    def figure(name):
        payload=base64.b64encode((ROOT/'figures'/f'{name}.png').read_bytes()).decode()
        return f'<figure><img alt="{html.escape(CAPTIONS[name])}" src="data:image/png;base64,{payload}"><figcaption>{html.escape(CAPTIONS[name])} Source: src/report.py → figures/{name}.png.</figcaption></figure>'
    intro=f'''<p class="kicker">UBA · Gravitational Waves and AI-Assisted Research · 2026</p><h1>Skipper-CCD background rejection with sideband patch anomalies</h1>
    <p class="byline">CODEX · gpt-6-astra · high effort · fixed chronological holdout</p>
    <p class="lead">On 22 held-out acquisitions, a 32×32-pixel Isolation Forest region mask retains <strong>{100*s['exposure_fraction']:.2f}% of effective exposure</strong> and reduces the reconstructed 4–7 electron background-rate proxy by <strong>{100*(1-s['background_ratio']):.2f}%</strong>. The largest improvement is concentrated in one module/quadrant. This result is relative to an active-geometry baseline, not the existing production cleaning pipeline.</p>'''
    metric_table=table(['Held-out metric','Baseline','Selected'],[
        ['Exposure [million pixel-days]',f"{s['baseline_exposure']/1e6:.6f}",f"{s['retained_exposure']/1e6:.6f}"],
        ['ROI clusters',s['baseline_roi_events'],s['final_roi_events']],
        ['ROI rate [events / million pixel-days]',f"{s['baseline_roi_background_rate']*1e6:.3f}",f"{s['final_roi_background_rate']*1e6:.3f}"],
        ['f_exp','1',f"{s['exposure_fraction']:.5f} [{ci['exposure_fraction'][0]:.5f}, {ci['exposure_fraction'][1]:.5f}]"],
        ['r_B','1',f"{s['background_ratio']:.5f} [{ci['background_ratio'][0]:.5f}, {ci['background_ratio'][1]:.5f}]"],
        ['F_equal','1',f"{s['F_equal']:.4f} [{ci['F_equal'][0]:.4f}, {ci['F_equal'][1]:.4f}]"],
        ['Conditional significance proxy sqrt(F_equal)','1',f"{s['sensitivity_proxy']:.4f}"]])
    content=intro+'<h2>Outcome and trade-off</h2>'+metric_table+'<p>Brackets are conditional 95% acquisition-block bootstrap intervals, not total systematic errors. No sensor-mass conversion is claimed; kg-days are unavailable.</p>'+figure('tradeoff')
    content+=f'''<h2>Dataset and baseline</h2><p>All {baseline['files']} files are usable under the declared geometry/calibration checks: 109 acquisition IDs, two modules, four HDUs per file, 515×3200 stored pixels. Active geometry is 512×3072 pixels/HDU. The full baseline is {baseline['exposure_pixel_days']/1e6:.3f} million pixel-days and {baseline['roi_events']:,} ROI clusters. Nominal EXPOSURE is zero; effective exposure accumulates during approximately 4.77 hours of readout. Raw FITS bytes are external and SHA-256 indexed in data/manifest.json.</p>
    <p>Subtract the median reverse overscan per row and fit the 0/1-electron peaks for gain. The ROI is <strong>3.5 ≤ reconstructed cluster charge &lt; 7.5 e⁻</strong>: integer bins 4,5,6,7. Clusters are 8-connected pixels above 0.7 e⁻; charge is their above-threshold sum. Accept a whole pre-reconstructed cluster according to its maximum-charge pixel’s tile. No reclustering is performed after masking.</p>'''+figure('calibration')+figure('spectra')
    transient=json.loads((ROOT/'results/acquisition114_check.json').read_text())['paired_acquisition_exclusion']
    content+=f'''<p><strong>Calibration anomaly:</strong> all four S99 gains drop to 170–181 ADU/e⁻ in acquisition114, compared with roughly 210–220 in adjacent images. The raw electron peaks shift too; the cause is unknown. Its end timestamp is 10 seconds later than the paired module. It passes the predeclared calibration rules and remains in the primary sample. Excluding the whole paired acquisition as a post-test diagnostic gives f_exp={transient['exposure_fraction']:.4f}, r_B={transient['background_ratio']:.4f}, F_equal={transient['F_equal']:.3f}; no primary selection was changed. See results/acquisition114_check.json.</p>'''
    content+='''<h2>Independent method selection</h2><p>The primary contribution is an unsupervised multivariate patch classifier. Its six inputs summarize low-charge sideband clusters, occupied pixels of high-charge clusters, negative tails, temporal persistence and local neighborhoods. It never receives target-event counts or positions. All clusters with charge 2.5–20 e⁻ are excluded from positive diagnostic channels, including their individual pixels.</p>
    <p>Three primary families were tested: dynamic robust scores, combined persistent/dynamic robust scores, and Isolation Forest. Static sideband occupancy and single-channel scores are comparison methods; none is a rewrite of the production masks. Robust PCA, Laplacian cosmic-ray cleaning and autoencoders were considered but not implemented: their extra assumptions or focus on individual events were not justified by this task. METHOD.md records these decisions and references.</p>'''
    candidates=[]
    for method in sorted(set(r['method'] for r in operating)):
        rr=[r for r in operating if r['method']==method and r['split']=='validation' and float(r['exposure_fraction'])>=.8]
        b=max(rr,key=lambda r:float(r['F_equal']))
        candidates.append([method,b['tile_size'],b['training_removed_target'],f"{float(b['exposure_fraction']):.3f}",f"{float(b['background_ratio']):.3f}",f"{float(b['F_equal']):.3f}"])
    content+=table(['Method','Tile','Train removal target','Validation f_exp','Validation r_B','Validation F'],candidates)
    content+=f'''<h2>Validation protocol</h2><p>Train: IDs22–86 (65 acquisitions). Validation: 87–108 (22). Test: 109–130 (22). Paired modules and all eight HDUs stay together. The data contain one paired end-timestamp discrepancy, so grouping uses shared image ID rather than exact timestamp matching. Hyperparameters are selected on validation only: maximize F_equal among primary candidates with f_exp ≥ 0.8. Thresholds are training-exposure-weighted score quantiles, applied absolutely to later images.</p>
    <p>The chosen Forest has 100 trees, 256 samples/tree, six features, seed 20260927 and threshold {f['threshold']:.9f}. Selection was committed as 1812e3f before opening test ROI results. Validation retained {f['exposure_fraction']:.4f} with r_B={f['background_ratio']:.4f}; test retains {s['exposure_fraction']:.4f} with r_B={s['background_ratio']:.4f}. Test operating curves and ablations are descriptive and did not alter selection.</p>'''+figure('temporal')+figure('representative')
    content+='<h2>Detector differences and spatial evidence</h2>'+table(['Module','HDU','ROI before','ROI after','f_exp','r_B'],[[g['module'],g['hdu'],g['baseline_roi_events'],g['final_roi_events'],f"{g['exposure_fraction']:.3f}",f"{g['background_ratio']:.3f}"] for g in groups])+figure('spatial')
    content+='''<p>S101 HDU0 supplies most of the baseline ROI excess and most of the rejection. Its retained fraction is much lower than the global retained fraction; a pooled 80% constraint is not an 80% constraint per quadrant. This needs to be acceptable to a future experimental signal model before adopting the cut. Sideband persistence between training halves is measured in results/persistence.csv; acquisition histories and dynamic feature ablations probe transient behavior.</p>'''
    content+='<h2>Checks, ablations, and negative results</h2>'+figure('ablations')
    ablation_rows=read_csv('ablations.csv')
    content+=table(['Forest descriptor','Split','f_exp','r_B','F_equal'],[[r['ablation'],r['split'],f"{float(r['exposure_fraction']):.4f}",f"{float(r['background_ratio']):.4f}",f"{float(r['F_equal']):.3f}"] for r in ablation_rows])
    content+='''<p><strong>Negative result:</strong> removing the negative-tail or neighborhood feature improves the matched-budget Forest on both validation and test; the six-feature descriptor is not established as optimal. Removing persistence or high-charge information worsens rejection. These diagnostic ablations were run after the primary configuration was frozen and the test was opened, so they are reported without replacing the final selection. A new independent dataset is needed to confirm a revised smaller descriptor.</p>'''
    stresses=read_csv('reconstruction_checks.csv')
    content+=table(['Fixed-mask reconstruction check','ROI before','ROI after','r_B','F_equal'],[[r['variation'],r['baseline_roi_events'],r['final_roi_events'],f"{float(r['background_ratio']):.4f}",f"{float(r['F_equal']):.3f}"] for r in stresses])
    erased=sum(r['erased_roi_events'] for r in v['real_roi_erasure'])
    shifted=v['shifted_mask_background_ratio_95']
    content+=f'''<p>Five synthetic/analytic unit checks pass. A separate graph-walk implementation checks connected components; arithmetic-series exposure and direct event loops reproduce the final counts and exposure. Removing {erased} actual ROI clusters from all eight HDUs of test acquisition109 leaves every diagnostic feature exactly unchanged. Isolated synthetic 4–7 e⁻ injections also leave the feature maps unchanged. Exposure-weighted simulated signal anchors survive with fraction {v['injected_uniform_point_acceptance']:.5f} (binomial standard error {v['injection_binomial_se']:.5f}), consistent with f_exp; this does not measure reconstruction efficiency.</p>
    <p>Randomly translating each HDU’s mask yields a median rate ratio {shifted[1]:.3f}, with a central 95% range [{shifted[0]:.3f}, {shifted[2]:.3f}], compared with the aligned mask’s {s['background_ratio']:.3f}. This is a spatial-alignment control, not a p-value: it preserves HDU-specific mask areas and those HDUs have different rates. Negative-tail-only rejection is weak; static sideband and robust scores underperform the selected Forest under the declared validation exposure constraint. These results and every tested point remain in CSV outputs.</p>'''
    content+='''<h2>Physical interpretation and limitations</h2><p>For N ROI events and exposure E, B=N/E, f_exp=E1/E0 and r_B=B1/B0. The auxiliary F_equal=f_exp/r_B does not replace the Pareto curve. For a spatially uniform signal proportional to pixel-days, unchanged efficiency and a known background in the Gaussian counting regime, S=sE and Z≈S/√N=s√(E/B), giving Z1/Z0=√F_equal. This is only a conditional sensitivity proxy. Readout-induced or collision-correlated signals can violate that exposure model.</p>
    <p>The rate normalization assumes a uniform readout clock and zero initial integration. The baseline lacks production masks; residual backgrounds, edge effects, merged clusters, readout transients and calibration-model imperfections remain. The study does not establish a physical background mechanism, true interaction-charge spectrum, mass exposure, extended-signal efficiency, or generalization to another detector/campaign. The paired four-acquisition bootstrap does not cover calibration, geometry or selection systematics, and may miss longer correlations. Reconstruction stress checks hold the trained mask fixed. Some calibration fits have reduced chi-square above unity; gain/noise metadata are published for scrutiny.</p>
    <h2>Reproducibility and provenance</h2><p>Set SKIPPER_DATA_DIR to the directory containing the 218 pp_proc FITS files, install requirements.txt in a Python 3.13 environment (observed: 3.13.5), then run:</p><pre>SKIPPER_DATA_DIR=/path/to/pp_proc bash CODEX/gpt-6-astra_high/reproduce.sh</pre>
    <p>The workflow verifies input hashes, reconstructs the baseline and catalogs from FITS, fits development models, regenerates the held-out evaluation, runs numerical/stress checks, and rebuilds every figure and this report. No FITS data or caches belong in Git. The data downloader accepts an owner-supplied Google Drive URL/ID; none was provided, so no public dataset link is invented. README.md gives full setup and data instructions.</p>
    <p>All scientific code was written for this run. Astropy reads FITS; SciPy fits calibration peaks and labels components; scikit-learn supplies Isolation Forest; NumPy performs array arithmetic; Matplotlib generates figures and the PDF. Parent MOSKITA code was read only for geometry/calibration/exposure conventions. No parent code is imported. Every figure is generated by src/report.py; PROVENANCE.md maps outputs to source functions, configuration, libraries and checks. The HTML embeds every figure and is portable as one file.</p>
    <p>Session turn_context and local configuration agree on gpt-6-astra, high effort; a more specific backend snapshot is unavailable. ORIGINAL_PROMPT.md is verbatim. This run supplies separate verification calculations but no independent human or different-model review. The repository started on an unborn Claude-named branch; Codex created its own branch and did not alter CLAUDE/.</p>
    <h2>Sources</h2><ul><li><a href="https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf">Liu, Ting &amp; Zhou (2008), Isolation Forest</a>, the selected anomaly-detection family.</li>
    <li><a href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.IsolationForest.html">scikit-learn 1.6 IsolationForest</a>, implementation and score interface.</li>
    <li><a href="https://arxiv.org/abs/0912.3599">Candès et al., Robust PCA</a> and <a href="https://arxiv.org/abs/astro-ph/0108003">van Dokkum, Laplacian cosmic-ray rejection</a>, considered alternatives.</li>
    <li><a href="https://arxiv.org/abs/2106.08347">SENSEI single-electron characterization</a>, motivation for exposure/readout caveats.</li>
    <li><a href="https://github.com/matiaszaldarriaga/GW-AI-course/blob/main/final-project.html">UBA course final project</a>, plus day4/day5 provenance and verification material, inspected at commit 75ba1e0.</li></ul>'''
    content+=f'<footer>Scientific code commit: {html.escape(s["git_commit"])}. Seed: {s["random_seed"]}. Quantitative values are read from generated JSON/CSV files.</footer>'
    style='''body{font:17px/1.65 system-ui,sans-serif;color:#23303c;background:#f5f6f8;margin:0}main{max-width:1100px;margin:auto;background:white;padding:48px 5vw}h1{font:600 38px/1.18 Georgia,serif;max-width:900px}h2{margin-top:2.4em;color:#194d6c;font-size:25px}p{max-width:1000px}.kicker,.byline,figcaption,footer{font-size:14px;color:#586775}.lead{font-size:21px;border-left:4px solid #23638c;padding-left:20px}table{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}th,td{text-align:left;padding:9px;border-bottom:1px solid #dce3e8}th{background:#eef3f6}figure{margin:28px 0}img{width:100%;height:auto}figcaption{line-height:1.45}pre{white-space:pre-wrap;background:#eef3f6;padding:16px}footer{border-top:1px solid #ccc;margin-top:40px;padding-top:20px}a{color:#17618b}@media(max-width:600px){main{padding:25px 16px}h1{font-size:29px}table{font-size:11px}th,td{padding:5px}}'''
    (ROOT/'report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Skipper-CCD sideband anomaly rejection</title><style>'+style+'</style></head><body><main>'+content+'</main></body></html>')
    # Portable presentation companion, generated with the same installed plotting library.
    with PdfPages(ROOT/'report.pdf',metadata={'Title':'Skipper-CCD background rejection','Author':'CODEX gpt-6-astra high','CreationDate':None,'ModDate':None}) as pdf:
        fig=plt.figure(figsize=(11.7,8.3));fig.text(.07,.88,'Skipper-CCD background rejection',fontsize=25,weight='bold')
        fig.text(.07,.82,'UBA - Gravitational Waves and AI-Assisted Research',fontsize=13)
        paras=[f"Held-out result: {100*s['exposure_fraction']:.2f}% of exposure retained; ROI rate reduced by {100*(1-s['background_ratio']):.2f}%. Counts: {s['baseline_roi_events']:,} to {s['final_roi_events']:,}. Rate: {s['baseline_roi_background_rate']*1e6:.2f} to {s['final_roi_background_rate']*1e6:.2f} per million pixel-days. F_equal = {s['F_equal']:.3f}.",
            'Method: 32 x 32 pixel patches; six ROI-blind sideband/persistence/noise descriptors; unsupervised Isolation Forest (100 trees, 256 samples/tree). Whole regions are masked; events are evaluated by their reconstructed maximum-charge anchor.',
            'Validation: 65 training acquisitions, 22 validation, 22 chronological held-out test; both modules and all HDUs stay together. Selection maximizes validation F_equal subject to at least 80% global exposure. Test is not retuned.',
            'Limitations: gain is concentrated in S101 HDU0. Baseline is active geometry, not production masks. Exposure assumes uniform readout timing; no kg-day claim or measured true-signal efficiency. Bootstrap intervals are conditional. Same-agent numerical verification, no independent scientist review.',
            'Reproduce: SKIPPER_DATA_DIR=/path/to/pp_proc bash CODEX/gpt-6-astra_high/reproduce.sh. Read report.html, METHOD.md and PROVENANCE.md for full definitions, sources and checks. No FITS data are stored in Git.']
        ypos=.72
        for para in paras:
            wrapped=textwrap.fill(para,110);fig.text(.07,ypos,wrapped,fontsize=11,va='top',linespacing=1.5);ypos-=.14
        pdf.savefig(fig);plt.close(fig)
        for name in ['tradeoff','spectra','representative','spatial','temporal','ablations','calibration']:
            fig=plt.figure(figsize=(11.7,8.3) if name!='spatial' else (8.3,11.7))
            ax=fig.add_axes([.04,.14,.92,.80]);ax.imshow(plt.imread(ROOT/'figures'/f'{name}.png'));ax.axis('off')
            fig.text(.05,.07,textwrap.fill(CAPTIONS[name],125 if name!='spatial' else 90),fontsize=9,va='center')
            pdf.savefig(fig);plt.close(fig)
    assets={f'figures/{name}.png':sha256(ROOT/'figures'/f'{name}.png') for name in CAPTIONS}
    (ROOT/'results/report_assets.json').write_text(json.dumps(assets,indent=2)+'\n')
    print('Generated 7 figures, standalone report.html, 8-page report.pdf and figure sidecars.')

if __name__=='__main__':main()
