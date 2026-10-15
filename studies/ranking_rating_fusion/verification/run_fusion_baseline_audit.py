"""Matched-information reconstruction baseline using the same observed gaps.
No oracle true reward is passed to reconstruction or policy optimization.
Synthetic true reward is used only to evaluate all resulting policies.
"""
import json
import hashlib
import time
import sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from scipy.special import expit, logsumexp
from run_fusion_experiments import CFG, HERE, OUT, make_cases, scalar_values, write_csv


def estimated_value(theta, rhat, f, ref):
    logits=theta[:,None]*f[None,:]+np.log(ref)[None,:]
    logp=logits-logsumexp(logits,axis=1,keepdims=True);p=np.exp(logp)
    return np.sum(p*rhat,axis=1)-np.sum(p*(logp-np.log(ref)),axis=1)


def optimize_estimated(rhat, f, ref):
    grid=np.linspace(-12.,12.,2401)
    logits=grid[:,None]*f[None,:]+np.log(ref)[None,:]
    logp=logits-logsumexp(logits,axis=1,keepdims=True);p=np.exp(logp)
    kl=np.sum(p*(logp-np.log(ref)),axis=1)
    vals=rhat@p.T-kl[None,:];best=np.argmax(vals,axis=1)
    lo=grid[np.maximum(0,best-1)];hi=grid[np.minimum(len(grid)-1,best+1)]
    ratio=(np.sqrt(5)-1)/2
    for _ in range(35):
        left=hi-ratio*(hi-lo);right=lo+ratio*(hi-lo)
        vl=estimated_value(left,rhat,f,ref);vr=estimated_value(right,rhat,f,ref)
        lo=np.where(vl<vr,left,lo);hi=np.where(vl>=vr,right,hi)
    theta=(lo+hi)/2
    current=estimated_value(theta,rhat,f,ref)
    return np.where(vals[np.arange(len(rhat)),best]>current,grid[best],theta)


def reconstruct(count,gap_sum):
    c0,c1,c2=count.T
    h11=c0+c2;h22=c1+c2;h12=-c2
    rhs1=-gap_sum[:,0]+gap_sum[:,2];rhs2=-gap_sum[:,1]-gap_sum[:,2]
    det=h11*h22-h12*h12;connected=det>0
    safe=np.where(connected,det,1)
    r1=(h22*rhs1-h12*rhs2)/safe;r2=(h11*rhs2-h12*rhs1)/safe
    rhat=np.stack([np.zeros(len(count)),r1,r2],axis=1)
    B=np.array([[-1.,0.],[0.,-1.],[1.,-1.]])
    for i in np.flatnonzero(~connected):
        H=B.T@(count[i,:,None]*B);rhs=B.T@gap_sum[i]
        rhat[i,1:]=np.linalg.lstsq(H,rhs,rcond=None)[0]
    return rhat,connected


def main():
    import csv
    start=time.time();rng=np.random.default_rng(CFG['seed']);make_cases(rng)
    original=list(csv.DictReader((OUT/'finite_sample.csv').open()))
    original_index={(r['case'],int(r['n']),float(r['rating_noise_sd']),float(r['lambda'])):r for r in original}
    baseline_config_path=HERE/'fusion_baseline_config.json'
    json.loads(baseline_config_path.read_text())
    f=np.array([0.,.7,1.]);ref=np.ones(3)/3;a,b=np.triu_indices(3,1);d=f[a]-f[b]
    rewards={'original_negative':np.array([0.,10.,4.]),'rescaled_negative':np.array([0.,3.,1.2]),
             'realizable_moderate':1.3*f,'realizable_saturated':8*f}
    reps=CFG['finite_sample']['replicates'];rows=[];max_error=0.;max_exact_error=0.
    for name in CFG['finite_sample']['instances']:
        r=rewards[name];gap=r[a]-r[b];truep=expit(gap)
        for sd in CFG['finite_sample']['rating_noise_sd']:
            for n in CFG['finite_sample']['sample_sizes']:
                count=rng.multinomial(n,np.ones(3)/3,size=reps);wins=rng.binomial(count,truep[None,:])
                gap_sum=count*gap[None,:]+rng.normal(size=(reps,3))*sd*np.sqrt(count)
                rhat,connected=reconstruct(count,gap_sum)
                if sd==0 and np.any(connected):
                    error=float(np.max(np.abs(rhat[connected]-(r-r[0]))));max_exact_error=max(max_exact_error,error)
                    assert error<1e-10
                tb=optimize_estimated(rhat,f,ref);vb=scalar_values(tb,f,r,ref)
                values={}
                for lam in CFG['finite_sample']['rating_weights']:
                    lo=np.full(reps,-12.);hi=np.full(reps,12.)
                    for _ in range(55):
                        mid=(lo+hi)/2
                        derivative=np.sum(d[None,:]*(count*(expit(mid[:,None]*d[None,:])+lam*mid[:,None]*d[None,:])-wins-lam*gap_sum),axis=1)/n
                        lo=np.where(derivative<0,mid,lo);hi=np.where(derivative>=0,mid,hi)
                    val=scalar_values((lo+hi)/2,f,r,ref);values[lam]=val
                    original_mean=float(original_index[(name,n,sd,lam)]['mean_value_minus_binary'])
                    error=abs(float(np.mean(val-values[0.]))-original_mean);max_error=max(max_error,error)
                    assert error<1e-10,'Regenerated observations differ from original sweep'
                    diff=vb-val;mean=float(np.mean(diff));se=float(np.std(diff,ddof=1)/np.sqrt(reps))
                    rows.append({'case':name,'n':n,'rating_noise_sd':sd,'compared_lambda':lam,
                                 'reconstruction_mean_value':float(np.mean(vb)),
                                 'reconstruction_minus_fusion_mean':mean,'paired_MC_standard_error':se,
                                 'paired_MC_95_lower':mean-1.96*se,'paired_MC_95_upper':mean+1.96*se,
                                 'connected_graph_fraction':float(np.mean(connected))})
        print(f'Reconstruction control {name} complete',flush=True)
    write_csv(OUT/'reconstruction_baseline.csv',rows)
    out={'scope':'same observed gap data; no true reward supplied to baseline',
         'comparison_rows':len(rows),'replicates_per_row':reps,'elapsed_seconds':time.time()-start,
         'max_regenerated_original_mean_error':max_error,
         'max_exact_reward_reconstruction_error_on_connected_graph':max_exact_error,
         'claim':'standard reward reconstruction is a strong control, not a new algorithm',
         'assertions_passed':True}
    manifest_path=OUT/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['baseline_audit']={**out,'configuration_file':baseline_config_path.name,
                              'configuration_sha256':hashlib.sha256(baseline_config_path.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':main()
