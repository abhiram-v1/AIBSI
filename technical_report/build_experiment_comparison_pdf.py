"""Build a five-page evidence-based comparison, with standalone Matplotlib plots.

Run `charts` with the project's Python3.10; run `pdf` with bundled ReportLab Python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'tmp/pdfs/experiment_comparison'
OUTPUT = ROOT / 'output/pdf/AIBSI_Three_Experiment_Comparison_2026-11-03.pdf'
GRAPH = ROOT / 'neural_comparison/graph_ntf_svm/outputs/full'
PAPER = ROOT / 'neural_comparison/paper_benchmark/outputs/full_174'
ADV = ROOT / 'neural_comparison/advanced_clean_spectral/outputs'
COLORS = {'navy':'#142D4E','teal':'#087F8C','blue':'#4779B8','orange':'#D88332',
          'purple':'#7458A3','gray':'#617083','light':'#EDF3F7','ink':'#182A3D'}


def read(path):
    return json.loads(Path(path).read_text())


def charts():
    import sys
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
    from threadpoolctl import threadpool_limits
    import joblib
    sys.path.insert(0,str(ROOT/'neural_comparison/graph_ntf_svm'))
    import core

    WORK.mkdir(parents=True,exist_ok=True)
    papers=read(PAPER/'summary.json');controls=read(PAPER/'matched_previous_controls.json')
    graph=read(GRAPH/'summary.json')['gntf']
    svm=read(ADV/'results/site_normalized_metrics.json')['test']
    eegpt=read(ADV/'deep_models/eegpt_attention/metrics.json')['test']
    blend_full=read(ADV/'deep_models/eegpt_attention/svm_blend_metrics.json')
    blend=blend_full['test']
    data=np.load(ROOT/'neural_comparison/graph_ntf_svm/outputs/prepared/dataset.npz')
    raw=[];cal=[]
    with threadpool_limits(limits=3):
        for repeat in range(3):
            a=np.full(174,-1,int);b=np.full(174,-1,int)
            for folder in sorted(GRAPH.glob(f'repeat_{repeat:02d}_*/gntf')):
                r=read(folder/'result.json');ix=np.asarray(r['test_indices'])
                model=joblib.load(folder/'model.joblib')
                z=model.features.transform(data['x'][ix],data['sites'][ix],data['direct'][ix])
                a[ix]=model.svm.predict(z);b[ix]=np.asarray(r['probabilities']).argmax(1)
            for pred,dest in [(a,raw),(b,cal)]:
                dest.append({'accuracy':accuracy_score(data['y'],pred),
                             'balanced_accuracy':balanced_accuracy_score(data['y'],pred),
                             'macro_f1':f1_score(data['y'],pred,average='macro'),
                             'ftd_recall':recall_score(data['y'],pred,labels=[1],average='macro')})
    diagnostic={k:{field:float(np.mean([r[field] for r in rows])) for field in rows[0]}
                for k,rows in [('raw',raw),('calibrated',cal)]}
    sources=[GRAPH/'summary.json',GRAPH/'oof_predictions.json',PAPER/'summary.json',
             PAPER/'matched_previous_controls.json',PAPER/'verification.json',
             ADV/'results/site_normalized_metrics.json',ADV/'deep_models/eegpt_attention/metrics.json',
             ADV/'deep_models/eegpt_attention/svm_blend_metrics.json']
    payload={'graph':graph,'papers':papers,'controls':controls,'svm':svm,'eegpt':eegpt,'blend':blend,
             'blend_selection':blend_full['selection'],'diagnostic':diagnostic,
             'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (WORK/'report_data.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#CCD5DE','axes.labelcolor':COLORS['ink'],
                         'xtick.color':COLORS['gray'],'ytick.color':COLORS['ink'],'savefig.facecolor':'white'})
    def save(fig,name):
        fig.savefig(WORK/f'{name}.png',dpi=220,bbox_inches='tight')
        fig.savefig(WORK/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig)
    names=['CP6 + Naive Bayes','Handcrafted + SVM','DICE-Net (adapted)','Microstate cascade','Graph-NTF + SVM','Direct spectral SVM','Tensor Train + SVM']
    vals=[papers[k] for k in ['li_cp6_gnb','rostamikia_hierarchical_svm','dice_net_3class','jiang_microstate_cascade']]+[controls[k] for k in ['gntf','direct','tt']]
    fig,ax=plt.subplots(figsize=(8.2,4.1),layout='constrained');yy=np.arange(7)
    for offset,key,color,label in [(-.17,'accuracy',COLORS['teal'],'Accuracy'),(.17,'balanced_accuracy',COLORS['blue'],'Balanced accuracy')]:
        bars=ax.barh(yy+offset,[100*v[key] for v in vals],height=.29,color=color,label=label)
        ax.bar_label(bars,fmt='%.1f',padding=4,fontsize=9)
    ax.set(yticks=yy,yticklabels=names,xlim=(0,100),xlabel='Score (%)');ax.invert_yaxis()
    ax.xaxis.grid(True,alpha=.15);ax.set_axisbelow(True);ax.legend(loc='lower right',frameon=False)
    save(fig,'matched_comparison')

    fig,axes=plt.subplots(1,2,figsize=(8,3.3),layout='constrained')
    for ax,key,title in zip(axes,['gntf','li_cp6_gnb'],['Graph-NTF + SVM','CP6 + Naive Bayes']):
        m=controls[key] if key=='gntf' else papers[key]
        cm=np.asarray(m['confusion_matrix']);fraction=cm/cm.sum(1,keepdims=True)
        ax.imshow(fraction,cmap='Blues',vmin=0,vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f'{cm[i,j]}\n{100*fraction[i,j]:.1f}%',ha='center',va='center',fontsize=10,
                        color='white' if fraction[i,j]>.55 else COLORS['ink'])
        ax.set(xticks=range(3),yticks=range(3),xticklabels=['AD','FTD','HC'],yticklabels=['AD','FTD','HC'],
               xlabel='Predicted class',ylabel='Actual class',title=title)
        ax.tick_params(length=0)
    save(fig,'matched_confusions')

    fig,ax=plt.subplots(figsize=(7.8,2.7),layout='constrained');xx=np.arange(3)
    for offset,key,color,label in [(-.18,'raw',COLORS['blue'],'Direct SVM decision'),(.18,'calibrated',COLORS['orange'],'Calibrated-probability argmax')]:
        values=[100*diagnostic[key][k] for k in ['accuracy','macro_f1','ftd_recall']]
        bars=ax.bar(xx+offset,values,.32,color=color,label=label);ax.bar_label(bars,fmt='%.1f',padding=3)
    ax.set(xticks=xx,xticklabels=['Accuracy','Macro F1','FTD recall'],ylim=(0,100),ylabel='Score (%)')
    ax.yaxis.grid(True,alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper right',frameon=False,fontsize=9)
    save(fig,'calibration_diagnostic')

    fig,ax=plt.subplots(figsize=(7.8,3.0),layout='constrained');xx=np.arange(3)
    for j,(key,color,label) in enumerate([('accuracy',COLORS['teal'],'Accuracy'),('balanced_accuracy',COLORS['blue'],'Balanced accuracy'),('macro_f1',COLORS['orange'],'Macro F1')]):
        bars=ax.bar(xx+(j-1)*.24,[100*m[key] for m in [svm,eegpt,blend]],.22,color=color,label=label)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    ax.set(xticks=xx,xticklabels=['Log-mel SVM','EEGPT + attention','SVM + EEGPT blend'],ylim=(0,100),ylabel='Score (%)')
    ax.yaxis.grid(True,alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper center',ncols=3,frameon=False,fontsize=9)
    save(fig,'fixed_split_blend')
    print('Created four Matplotlib figures and verified source-data snapshot',flush=True)


def pdf():
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor,white
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph,Table,TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.utils import ImageReader
    from PIL import Image

    d=read(WORK/'report_data.json');p=d['papers'];g=d['graph']['mean'];ct=d['controls']
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    fontdir=Path('C:/Windows/Fonts')
    pdfmetrics.registerFont(TTFont('Body',str(fontdir/'arial.ttf')))
    pdfmetrics.registerFont(TTFont('BodyBold',str(fontdir/'arialbd.ttf')))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
    c=canvas.Canvas(str(OUTPUT),pagesize=A4)
    c.setTitle('AIBSI - Comparison of Three EEG Classification Experiments')
    c.setAuthor('AIBSI Project')
    W,H=A4;M=38;CW=W-2*M
    styles={
        'body':ParagraphStyle('body',fontName='Body',fontSize=10,leading=14,textColor=HexColor(COLORS['ink'])),
        'small':ParagraphStyle('small',fontName='Body',fontSize=8.3,leading=11,textColor=HexColor(COLORS['gray'])),
        'tiny':ParagraphStyle('tiny',fontName='Body',fontSize=7.3,leading=10,textColor=HexColor(COLORS['gray'])),
        'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.4,leading=11,textColor=HexColor(COLORS['ink'])),
        'headcell':ParagraphStyle('headcell',fontName='BodyBold',fontSize=8.2,leading=10.5,textColor=white)}
    def para(text,y,style='body',x=M,width=CW):
        obj=Paragraph(text,styles[style]);_,height=obj.wrap(width,H)
        obj.drawOn(c,x,y-height);return y-height
    def title(number,label,heading,sub):
        c.setFillColor(HexColor(COLORS['navy']));c.rect(0,H-13,W,13,fill=1,stroke=0)
        c.setFont('BodyBold',9);c.drawString(M,H-37,'AIBSI  /  EXPERIMENT COMPARISON')
        c.setFont('Body',8);c.setFillColor(HexColor(COLORS['gray']));c.drawRightString(W-M,H-37,'03 NOV 2026')
        c.setFillColor(HexColor(COLORS['teal']));c.setFont('BodyBold',8);c.drawString(M,H-63,label.upper())
        c.setFillColor(HexColor(COLORS['navy']));c.setFont('BodyBold',23);c.drawString(M,H-93,heading)
        y=para(sub,H-108,'small')
        c.setStrokeColor(HexColor('#DCE5ED'));c.line(M,38,W-M,38)
        c.setFillColor(HexColor(COLORS['gray']));c.setFont('Body',7.5)
        c.drawString(M,25,'AD / FTD / HC  |  Participant-level research evaluation')
        c.drawRightString(W-M,25,f'{number} / 5')
        return y-17
    def section(text,y):
        c.setFont('BodyBold',11);c.setFillColor(HexColor(COLORS['navy']));c.drawString(M,y-11,text)
        return y-22
    def table(headers,rows,widths,y,highlight=None):
        cells=[[Paragraph(str(v),styles['headcell']) for v in headers]]
        cells += [[Paragraph(str(v),styles['cell']) for v in row] for row in rows]
        t=Table(cells,colWidths=widths,hAlign='LEFT')
        commands=[('BACKGROUND',(0,0),(-1,0),HexColor(COLORS['navy'])),('VALIGN',(0,0),(-1,-1),'TOP'),
                  ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),7),
                  ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
                  ('LINEBELOW',(0,0),(-1,0),.5,HexColor(COLORS['navy']))]
        for i in range(1,len(cells)):
            commands.append(('BACKGROUND',(0,i),(-1,i),HexColor('#F1F6F9' if i%2 else '#FFFFFF')))
            commands.append(('LINEBELOW',(0,i),(-1,i),.35,HexColor('#DFE7ED')))
        if highlight is not None:commands.append(('BACKGROUND',(0,highlight),(-1,highlight),HexColor('#DFF1EF')))
        t.setStyle(TableStyle(commands));_,height=t.wrap(CW,H);t.drawOn(c,M,y-height);return y-height-10
    def image(name,y,width=CW):
        path=WORK/f'{name}.png'
        with Image.open(path) as img:iw,ih=img.size
        height=width*ih/iw;c.drawImage(ImageReader(str(path)),M,y-height,width,height,mask='auto');return y-height-10
    def callout(text,y,color='teal'):
        obj=Paragraph(text,styles['body']);_,height=obj.wrap(CW-22,H)
        c.setFillColor(HexColor('#EDF6F5' if color=='teal' else '#FFF4E8'))
        c.roundRect(M,y-height-19,CW,height+19,6,fill=1,stroke=0)
        c.setFillColor(HexColor(COLORS[color]));c.rect(M,y-height-19,3,height+19,fill=1,stroke=0)
        obj.drawOn(c,M+12,y-height-9);return y-height-31
    def pct(v):return f'{100*v:.1f}%'
    def done(y):
        if y<48:raise RuntimeError(f'Page content runs into footer: y={y}')
        c.showPage()

    # Page 1: three experiments and their actual evaluation scope.
    y=title(1,'Overview','Three experiments, one task',
            'Comparison of the previous Graph-NTF run, the current paper-method adaptations, and the earlier log-mel SVM + EEGPT blend. All scores below are our experiments, not published paper accuracies.')
    y=callout('<b>Main finding:</b> CP6 + Naive Bayes reached <b>60.3% accuracy</b> on the matched 174-person evaluation. The SVM + EEGPT blend reached <b>70.4%</b> on a separate 27-person fixed test split; that score is exploratory and cannot be ranked directly against cross-validation.',y)
    y=section('Dataset and prediction target',y)
    y=para('The merged cohort contains <b>174 people: 71 AD, 42 FTD and 61 healthy controls</b>, from OpenNeuro ds004504 and usable BrainLat EEG. Every result predicts one of three classes for a person. Windows are repeated measurements, not additional independent patients.',y)-14
    rows=[['A - Graph-NTF + SVM','174 people; 3 repeats x 5 outer folds; 3 inner folds','53.4%','47.2%','0.8%'],
          ['B - CP6 + Naive Bayes<br/>(best adaptation)','174 people; 1 x 5 shared outer folds; 3 inner folds','60.3%','56.5%','31.0%'],
          ['C - Log-mel SVM + EEGPT','121 train / 26 validation / 27 test; fixed split','70.4%','67.6%','50.0%']]
    y=table(['Experiment','Evaluation','Accuracy','Balanced<br/>accuracy','FTD<br/>recall'],rows,[136,190,64,70,CW-460],y)
    y=para('A reports the mean across three repeats. Page 2 uses only A\'s first repeat to match B exactly. C evaluates 27 test participants, including only six with FTD. Balanced accuracy is the average of AD, FTD and HC recall. FTD recall is the proportion of true FTD cases correctly identified.',y,'small')-17
    y=section('What changed between the experiments?',y)
    rows=[['A','Nonnegative electrode-frequency-statistic tensor; graph-smoothed CP factors; site centering; SVM; probability calibration.'],
          ['B','Four literature adaptations: CP6 + Naive Bayes, handcrafted SVM cascade, DICE-Net and microstate CNN cascade. Each evaluated on held-out people.'],
          ['C','Site-normalized log-mel SVM combined with a partially fine-tuned EEGPT participant model using validation-selected probability blending.']]
    y=table(['Run','Method'],rows,[35,CW-35],y)
    y=para('Prepared on 03 November 2026. A and B are identified by their saved experiment outputs rather than an assumed wall-clock run date. No independent external cohort has been evaluated.',y,'small')
    done(y)

    # Page 2: directly matched cohort and folds.
    y=title(2,'Matched cross-validation','Graph-NTF versus paper methods',
            'Same 174 people, the same five outer test memberships, and exactly one held-out prediction per person. Previous controls use repeat 0 only. Representations and training procedures still differ.')
    y=image('matched_comparison',y)
    keys=['li_cp6_gnb','rostamikia_hierarchical_svm','dice_net_3class','jiang_microstate_cascade']
    labels=['CP6 + Naive Bayes','Handcrafted + SVM','DICE-Net adaptation','Microstate cascade']
    rows=[]
    for name,key in zip(labels,keys):
        m=p[key];rows.append([name,pct(m['accuracy']),pct(m['balanced_accuracy']),pct(m['macro_f1']),pct(m['per_class']['FTD']['recall'])])
    for name,key in [('Graph-NTF + SVM','gntf'),('Direct spectral SVM','direct'),('Tensor Train + SVM','tt')]:
        m=ct[key];rows.append([name,pct(m['accuracy']),pct(m['balanced_accuracy']),pct(m['macro_f1']),pct(m['per_class']['FTD']['recall'])])
    y=table(['Method','Accuracy','Balanced acc.','Macro F1','FTD recall'],rows,[185,78,88,82,CW-433],y,highlight=1)
    y=callout('<b>CP6 gained 12 correct predictions:</b> 105/174 versus 93/174 for Graph-NTF on these folds (+6.9 percentage points). The microstate cascade has higher FTD recall (45.2%) but lower overall accuracy (41.4%); catching more FTD cases alone does not imply a better overall classifier.',y)
    done(y)

    # Page 3: exact FTD failure and diagnostic replay.
    y=title(3,'FTD and decision behaviour','Why accuracy alone is insufficient',
            'The top panels compare the same 174-person outer predictions. Cells show counts and percentages within the actual class; diagonal cells are correct predictions.')
    y=image('matched_confusions',y)
    y=para('<b>Matched five-fold FTD result:</b> Graph-NTF classified 0/42 FTD participants correctly; CP6 classified 13/42. CP6 also improved AD from 48/71 to 53/71, while HC decreased from 45/61 to 39/61.',y)-15
    y=section('Calibration diagnostic: same saved Graph-NTF models',y)
    y=image('calibration_diagnostic',y)
    y=para('The lower chart averages all three Graph-NTF repeats. Direct SVM decisions gave <b>26.2% FTD recall</b>; calibrated-probability argmax gave <b>0.8%</b>. Accuracy changed from 52.7% to 53.4%, while macro F1 fell from 49.2% to 41.1%. This is a diagnostic replay, not a newly tuned improvement.',y)-12
    y=callout('<b>Decision-stage mismatch:</b> the original search selected SVMs using direct class predictions, but final reporting used calibrated-probability argmax. Future tuning should select the complete final decision rule inside training folds and track FTD precision, recall and errors on AD/HC.',y,color='orange')
    done(y)

    # Page 4: fixed split, kept separate.
    y=title(4,'Separate fixed-split benchmark','Log-mel SVM + EEGPT blend',
            'Exploratory result on 27 held-out participants: 11 AD, 6 FTD and 10 HC. The full cohort split was 121 training / 26 validation / 27 test participants.')
    y=image('fixed_split_blend',y)
    rows=[]
    for name,m,count in [('Log-mel SVM',d['svm'],18),('EEGPT + attention',d['eegpt'],17),('SVM + EEGPT blend',d['blend'],19)]:
        rows.append([name,f'{count}/27',pct(m['accuracy']),pct(m['macro_f1']),pct(m['per_class']['FTD']['recall'])])
    y=table(['Model','Correct','Accuracy','Macro F1','FTD recall'],rows,[185,70,85,85,CW-425],y,highlight=3)
    y=section('How the two models were combined',y)
    y=para('The spectral branch used moderate-cleaning log-mel features, training-fitted site normalization, 20 PCA components and a linear SVM (C = 0.1). The EEGPT branch used pretrained EEG representations, participant window attention and partial fine-tuning.',y)-10
    y=para('Validation selected <b>50% SVM + 50% EEGPT</b> probability fusion after temperature scaling (SVM temperature 0.5; EEGPT temperature 0.75). The selected blend reached 73.1% validation accuracy and 71.0% validation macro F1.',y)-12
    y=callout('<b>What improved:</b> the blend corrected one additional test participant versus SVM alone. HC recall increased from 70% (7/10) to 80% (8/10). AD stayed at 8/11 and FTD stayed at 3/6. The gain was in healthy-control recognition, not FTD recognition.',y)
    y=para('<b>Comparison limit:</b> this 70.4% result comes from one small test split inspected during multiple development iterations. It is not an estimate from nested cross-validation and does not establish that the blend outperforms CP6 or Graph-NTF on unseen cohorts.',y,'small')
    done(y)

    # Page 5: interpretation, limitations and evidence.
    y=title(5,'Conclusions and evidence','What we can claim now',
            'The report preserves the difference between observed scores, interpretable representations and evidence of generalization.')
    y=section('Interpretability',y)
    rows=[['Graph-NTF + SVM','Electrode, frequency and statistic profiles, plus participant activations. Graph regularization encourages spatial smoothness; it does not prove anatomical causality.'],
          ['CP6 + Naive Bayes','Six electrode-frequency components and participant strengths. Component plots, stability and class contributions still need a dedicated interpretation analysis.'],
          ['SVM + EEGPT blend','Defined spectral inputs plus a learned transformer representation. The blend has no shared small set of CP components explaining both branches.']]
    y=table(['Pipeline','What is inspectable'],rows,[145,CW-145],y)
    y=section('Limits that materially affect the comparison',y)
    notes=[
        '<b>Optimization:</b> 7/15 Graph-NTF outer fits and 3/5 CP6 outer fits had no optimizer-converged initialization within their frozen budgets. Convergence was not assumed.',
        '<b>Adaptations:</b> the paper benchmark changes source-cohort details and some feature/architecture choices. DICE has a three-class head and corrected batch semantics; the SVM and microstate arms are documented reimplementations, not exact paper replications.',
        '<b>Data:</b> both recent benchmarks inherit moderate preprocessing without explicit EEGLAB boundary-window rejection. This signal-quality issue is separate from participant train/test leakage.',
        '<b>Validation:</b> the 174-person cohort has been explored repeatedly. The fixed-split blend is especially exploratory. No method has independent external validation.',
        '<b>Verification:</b> the paper benchmark checked input/code hashes and disjoint participant memberships; all 20 saved fold models reproduced their held-out decisions.'
    ]
    for text in notes:y=para(text,y,'small')-8
    y=section('Next controlled experiment',y)
    y=para('Prioritize FTD recall together with macro F1 and false positives. Select the complete decision/calibration rule inside training folds; improve tensor convergence; then vary representation, graph strength and classifier one at a time on shared participant splits.',y,'small')-14
    y=section('Source artifacts (relative to C:\\Projects\\AIBSI)',y)
    references=[
        'neural_comparison/graph_ntf_svm/outputs/full/summary.json',
        'neural_comparison/graph_ntf_svm/outputs/full/oof_predictions.json',
        'neural_comparison/paper_benchmark/outputs/full_174/summary.json',
        'neural_comparison/paper_benchmark/outputs/full_174/matched_previous_controls.json',
        'neural_comparison/paper_benchmark/outputs/full_174/verification.json',
        'neural_comparison/advanced_clean_spectral/outputs/deep_models/eegpt_attention/svm_blend_metrics.json',
        'neural_comparison/advanced_clean_spectral/outputs/results/site_normalized_metrics.json',
        'neural_comparison/advanced_clean_spectral/outputs/deep_models/eegpt_attention/metrics.json'
    ]
    for i,path in enumerate(references,1):
        y=para(f'{i}. '+path.replace('/','/<wbr/>'),y,'tiny')-3
    y=para('Full method departures: neural_comparison/paper_benchmark/README.md. Paper references: Li et al. (2024), doi:10.1093/braincomms/fcae227; Rostamikia et al. (2024), doi:10.1007/s11571-024-10152-7; DICE-Net (2023), doi:10.1109/ACCESS.2023.3294618; Jiang et al. (2026), doi:10.3390/bios16050258.',y-5,'tiny')
    done(y)
    c.save()
    print(OUTPUT,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['charts','pdf'])
    args=parser.parse_args()
    charts() if args.command=='charts' else pdf()
