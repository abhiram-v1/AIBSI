"""Participant-level label associations, multiplicity control and severity checks."""
import json
import numpy as np
from scipy.stats import mannwhitneyu,rankdata,t as student_t
from spectral_data import HERE,DESCRIPTORS,CHANNELS,REGIONS,dump

CONTRASTS={'dementia_vs_HC':(['HC'],['AD','FTD']),'AD_vs_HC':(['HC'],['AD']),
           'FTD_vs_HC':(['HC'],['FTD']),'AD_vs_FTD':(['FTD'],['AD'])}


def fdr(p):
    p=np.asarray(p);order=np.argsort(p);q=np.empty(len(p))
    q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    return q


def correlate(x,y):
    a=x-x.mean(0);b=y-y.mean();den=np.sqrt((a*a).sum(0)*(b*b).sum())
    return np.divide(b@a,den,out=np.zeros(x.shape[1]),where=den>1e-15)


def adjusted_label_test(x,y,age,sex):
    design=np.column_stack([np.ones(len(y)),y,age-age.mean(),sex])
    pinv=np.linalg.pinv(design);coef=pinv@x;res=x-design@coef;df=len(y)-np.linalg.matrix_rank(design)
    variance=(res*res).sum(0)/df;se=np.sqrt(np.maximum(variance*np.linalg.pinv(design.T@design)[1,1],0))
    stat=np.divide(coef[1],se,out=np.zeros(x.shape[1]),where=se>1e-12)
    return coef[1],2*student_t.sf(np.abs(stat),df)


def tests(x,mask0,mask1,names,age,sex):
    a=x[mask0];b=x[mask1];u,p=mannwhitneyu(b,a,axis=0,alternative='two-sided')
    n=len(a)+len(b);sd=np.sqrt(((len(a)-1)*a.var(0,ddof=1)+(len(b)-1)*b.var(0,ddof=1))/(n-2))
    g=np.divide(b.mean(0)-a.mean(0),sd,out=np.zeros(x.shape[1]),where=sd>1e-12)*(1-3/(4*n-9))
    subset=mask0|mask1;label=mask1[subset].astype(float);r=correlate(x[subset],label)
    coef,adjusted_p=adjusted_label_test(x[subset],label,age[subset],sex[subset])
    q=fdr(p);adjusted_q=fdr(adjusted_p)
    return [{'feature':name,'group0_n':len(a),'group1_n':len(b),'group0_median':float(np.median(a[:,i])),
             'group1_median':float(np.median(b[:,i])),'hedges_g_group1_minus_group0':float(g[i]),
             'cliff_delta':float(2*u[i]/(len(a)*len(b))-1),'point_biserial_r':float(r[i]),
             'mann_whitney_p':float(p[i]),'fdr_q':float(q[i]),'age_sex_adjusted_difference':float(coef[i]),
             'adjusted_p':float(adjusted_p[i]),'adjusted_fdr_q':float(adjusted_q[i])} for i,name in enumerate(names)]


def main():
    out=HERE/'outputs';prep=out/'prepared';ps=json.loads((prep/'plan.json').read_text())['participants']
    arrays=dict(np.load(prep/'features.npz'));names=json.loads((prep/'feature_names.json').read_text())
    dx=np.array([p['label'] for p in ps]);age=np.array([p['age'] for p in ps]);sex=np.array([p['sex']=='M' for p in ps],float)
    results={'scope':'exploratory participant-level associations; categorical labels tested as explicit contrasts, never numeric0/1/2 correlation',
             'fdr':'BH correction within each declared family and contrast; unadjusted rank test and age/sex-adjusted OLS reported separately',
             'families':{}}
    families={'global':(arrays['global_summary'],DESCRIPTORS),'regional':(arrays['regional_summary'].reshape(88,-1),
              [f'{region}:{desc}' for region in REGIONS for desc in DESCRIPTORS]),'channel_summary':(arrays['summary'],names['summary']),
              'frequency':(np.concatenate([arrays['absolute_spectrum'],arrays['relative_spectrum']],axis=1),
                           names['absolute_spectrum']+names['relative_spectrum'])}
    for family,(x,nn) in families.items():
        results['families'][family]={}
        for name,(group0,group1) in CONTRASTS.items():
            results['families'][family][name]=tests(x,np.isin(dx,group0),np.isin(dx,group1),nn,age,sex)
    dump(out/'associations.json',results)
    # Within-patient severity avoids the all-HC-MMSE30 group difference.
    mask=dx!='HC';mmse=np.array([p['mmse'] for p in ps])[mask];rank_y=rankdata(mmse)
    design=np.column_stack([np.ones(mask.sum()),rankdata(age[mask]),sex[mask],dx[mask]=='AD'])
    residual_y=rank_y-design@(np.linalg.pinv(design)@rank_y);df=int(mask.sum()-np.linalg.matrix_rank(design)-1)
    severity=[];xx=np.concatenate([arrays['global_summary'],arrays['summary']],axis=1)
    nn=[f'global:{d}' for d in DESCRIPTORS]+names['summary'];rank_x=rankdata(xx[mask],axis=0)
    r=correlate(rank_x,rank_y);residual_x=rank_x-design@(np.linalg.pinv(design)@rank_x)
    partial=correlate(residual_x,residual_y)
    p=2*student_t.sf(np.abs(partial)*np.sqrt(df/np.maximum(1-partial*partial,1e-12)),df);q=fdr(p)
    for i,name in enumerate(nn):severity.append({'feature':name,'patient_n':int(mask.sum()),'spearman_mmse':float(r[i]),
        'partial_rank_correlation':float(partial[i]),'partial_p':float(p[i]),'fdr_q':float(q[i])})
    dump(out/'severity_associations.json',{'scope':'AD+FTD only; partial rank correlation adjusted for ranked age,sex,and AD-versus-FTD; not causal',
                                        'tests':severity})
    for contrast in CONTRASTS:
        strongest=sorted(results['families']['global'][contrast],key=lambda v:v['fdr_q'])[:5]
        print(contrast,[(v['feature'],round(v['point_biserial_r'],3),round(v['fdr_q'],5)) for v in strongest],flush=True)
    print('Participant-level association and severity analyses saved.',flush=True)


if __name__=='__main__':main()
