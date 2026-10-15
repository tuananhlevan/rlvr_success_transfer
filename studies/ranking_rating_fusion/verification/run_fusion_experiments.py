"""CPU finite-action experiments. Run with the probml environment's Python.

No network, language-model training, or sample-wise reward oracle in fitted losses.
Population evaluation uses a known synthetic reward by design. Every configured
case is retained. Finite-sample weights use the same random observations.
"""
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fusion-mpl-config')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.special import expit, logsumexp
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'fusion_sprint1'
CONFIG_PATH = HERE / 'fusion_experiment_config.json'
CFG = json.loads(CONFIG_PATH.read_text())


def value_and_gradient(theta, f, r, reference, tau=1.):
    logits = np.log(reference) + f @ theta
    logp = logits - logsumexp(logits)
    p = np.exp(logp)
    centered_f = f - p @ f
    residual = r - tau * (f @ theta)
    gradient = centered_f.T @ (p * residual)
    value = p @ r - tau * (p @ (logp - np.log(reference)))
    return float(value), gradient, p


def pair_design(k, reference, source):
    a, b = np.triu_indices(k, 1)
    q = np.ones(len(a)) if source == 'uniform' else reference[a]*reference[b]
    return a, b, q/q.sum()


def loss_terms(theta, X, gap, q, lam, tau=1.):
    z = tau*(X @ theta)
    p = expit(z)
    truep = expit(gap)
    residual = z-gap
    loss = q @ (np.logaddexp(0.,z)-truep*z+lam*residual**2/2)
    gradient = tau * X.T @ (q*(p-truep+lam*residual))
    hessian = tau**2 * X.T @ ((q*(p*(1-p)+lam))[:,None]*X)
    return float(loss), gradient, hessian


def fit_population(X,gap,q,lam,start=None):
    theta = np.zeros(X.shape[1]) if start is None else start.copy()
    converged = False
    for iteration in range(150):
        loss,g,H = loss_terms(theta,X,gap,q,lam)
        if np.max(np.abs(g))<CFG['population']['gradient_tolerance']:
            converged = True
            break
        direction = np.linalg.solve(H+1e-13*np.eye(len(theta)),-g)
        dot = g @ direction
        step = 1.
        for _ in range(45):
            trial = theta+step*direction
            if loss_terms(trial,X,gap,q,lam)[0] <= loss+1e-4*step*dot+1e-14:
                theta=trial
                break
            step *= .5
        else:
            break
    loss,g,H=loss_terms(theta,X,gap,q,lam)
    return theta, {'converged':converged or np.max(np.abs(g))<1e-9,
                   'gradient_inf':float(np.max(np.abs(g))), 'iterations':iteration+1}


def write_csv(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def make_cases(rng):
    cases=[]
    for k in CFG['population']['actions']:
        # A centered orthonormal basis; nested ranks share the same feature draw.
        raw=rng.normal(size=(k,k-1));raw-=raw.mean(axis=0)
        Q=np.linalg.qr(raw)[0]
        ranks=sorted(set([1,min(3,k-1),k-1]))
        random_r=rng.normal(size=k)
        for rank in ranks:
            f=Q[:,:rank]
            for refname in CFG['population']['references']:
                reference=np.ones(k)/k if refname=='uniform' else np.exp(np.linspace(-1.5,1.5,k))
                reference=reference/reference.sum()
                for source in CFG['population']['pair_sources']:
                    for reward_range in CFG['population']['reward_ranges']:
                        for mode in CFG['population']['reward_modes']:
                            if mode=='random':base=random_r.copy()
                            elif mode=='nonmonotone':
                                s=f[:,0];base=-(s-np.median(s))**2
                            else:base=f @ np.linspace(.3,1.1,rank)
                            r=(base-base.min())*reward_range/(base.max()-base.min())
                            name=f'K{k}_d{rank}_{refname}_{source}_range{reward_range:g}_{mode}'
                            cases.append({'id':name,'k':k,'rank':rank,'refname':refname,
                                          'source':source,'range':reward_range,'mode':mode,
                                          'f':f,'r':r,'reference':reference})
    cases.insert(0,{'id':'original_negative','k':3,'rank':1,'refname':'uniform',
                    'source':'reference','range':10.,'mode':'handcrafted',
                    'f':np.array([[0.],[.7],[1.]]),'r':np.array([0.,10.,4.]),
                    'reference':np.ones(3)/3})
    return cases


def population_study(rng):
    cases=make_cases(rng);rows=[];diagnostics=[]
    for idx,c in enumerate(cases):
        f,r,ref=c['f'],c['r'],c['reference']
        a,b,q=pair_design(c['k'],ref,c['source']);X=f[a]-f[b];gap=r[a]-r[b]
        tB,infoB=fit_population(X,gap,q,0.)
        jB,gJ,pB=value_and_gradient(tB,f,r,ref)
        _,gR,H=loss_terms(tB,X,gap,q,0.)
        gR=X.T @ (q*(X@tB-gap))
        sensitivity=float(-gJ @ np.linalg.solve(H,gR))
        small_t,small_info=fit_population(X,gap,q,CFG['population']['small_weight'],tB)
        fd=(value_and_gradient(small_t,f,r,ref)[0]-jB)/CFG['population']['small_weight']
        diagnostics.append({'case':c['id'],'sensitivity':sensitivity,'finite_difference':fd,
                            'sign_matched':bool(sensitivity*fd>0) if abs(sensitivity)>1e-5 else None,
                            'gradient_inf':infoB['gradient_inf']})
        for lam in CFG['population']['rating_weights']:
            t,info=fit_population(X,gap,q,lam,tB)
            J,_,p=value_and_gradient(t,f,r,ref)
            loss,_,_=loss_terms(t,X,gap,q,0.)
            mse=float(q @ (X@t-gap)**2)
            rows.append({'case':c['id'],'actions':c['k'],'rank':c['rank'],
                         'reference':c['refname'],'pair_source':c['source'],'reward_range':c['range'],
                         'reward_mode':c['mode'],'lambda':lam,'value':J,'value_minus_binary':J-jB,
                         'rating_gap_mse':mse,'binary_CE':loss,'transfer_sensitivity':sensitivity,
                         'small_weight_value_derivative':fd,
                         'small_weight_sign_matched':bool(sensitivity*fd>0) if abs(sensitivity)>1e-5 else None,
                         'gradient_inf':info['gradient_inf'],'converged':info['converged'],
                         'iterations':info['iterations']})
        if (idx+1)%25==0:print(f'Population cases {idx+1}/{len(cases)}',flush=True)
    write_csv(OUT/'population.csv',rows)
    controls=[x for x in rows if x['reward_mode']=='realizable']
    assert max(abs(x['value_minus_binary']) for x in controls)<1e-6
    assert all(x['converged'] for x in rows), 'Nonconverged cases require review; raw outputs retained'
    return rows,diagnostics


def scalar_values(theta,f,r,ref):
    logits=np.asarray(theta)[:,None]*f[None,:]+np.log(ref)[None,:]
    logp=logits-logsumexp(logits,axis=1,keepdims=True);p=np.exp(logp)
    return p@r-np.sum(p*(logp-np.log(ref)),axis=1)


def scalar_moments(theta,f,r,ref,lam,noise_sd):
    a,b=np.triu_indices(len(f),1);d=f[a]-f[b];gap=r[a]-r[b]
    p=expit(theta*d);truep=expit(gap);q=np.ones(len(d))/len(d)
    mean_score=d*(p-truep+lam*(theta*d-gap))
    H=float(q @ (d*d*(p*(1-p)+lam)))
    T=float(q @ (d**3*p*(1-p)*(1-2*p)))
    h_i=d*d*(p*(1-p)+lam)
    # Conditional score variance plus between-pair mean-score variance.
    C=float(q @ (d*d*(truep*(1-truep)+lam*lam*noise_sd**2)+mean_score**2))
    Egh=float(q @ (mean_score*h_i))
    sigma=C/H**2
    bias=Egh/H**2-T*C/(2*H**3)
    h=1e-4
    J=float(scalar_values(np.array([theta]),f,r,ref)[0])
    Jp=float((scalar_values(np.array([theta+h]),f,r,ref)[0]-scalar_values(np.array([theta-h]),f,r,ref)[0])/(2*h))
    Jpp=float((scalar_values(np.array([theta+h]),f,r,ref)[0]-2*J+scalar_values(np.array([theta-h]),f,r,ref)[0])/h**2)
    V=-(Jp*bias+.5*Jpp*sigma)
    return {'theta_population':theta,'J_population':J,'H':H,'T':T,'C':C,
            'asymptotic_parameter_variance':sigma,'first_order_estimator_bias':bias,
            'J_prime':Jp,'J_second':Jpp,'expected_regret_1_over_n_coefficient':V}


def sample_study(rng):
    f=np.array([0.,.7,1.]);ref=np.ones(3)/3
    rewards={'original_negative':np.array([0.,10.,4.]),
             'rescaled_negative':np.array([0.,3.,1.2]),
             'realizable_moderate':1.3*f,
             'realizable_saturated':8.*f}
    a,b=np.triu_indices(3,1);d=f[a]-f[b]
    rows=[];coefficients=[];reps=CFG['finite_sample']['replicates']
    lo0,hi0=CFG['finite_sample']['parameter_bounds']
    for name in CFG['finite_sample']['instances']:
        r=rewards[name];gap=r[a]-r[b];truep=expit(gap)
        # Scalar global value evaluation by dense bracket plus bounded local solves.
        grid=np.linspace(lo0,hi0,2401);vg=scalar_values(grid,f,r,ref);best=np.argmax(vg)
        left=grid[max(0,best-1)];right=grid[min(len(grid)-1,best+1)]
        oracle=minimize_scalar(lambda t:-scalar_values(np.array([t]),f,r,ref)[0],bounds=(left,right),method='bounded')
        Jopt=max(float(vg[best]),float(-oracle.fun))
        pop={}
        for sd in CFG['finite_sample']['rating_noise_sd']:
            for lam in CFG['finite_sample']['rating_weights']:
                t,_=fit_population(d[:,None],gap,np.ones(3)/3,lam)
                coeff=scalar_moments(float(t[0]),f,r,ref,lam,sd)
                coeff.update({'case':name,'rating_noise_sd':sd,'lambda':lam,'restricted_regret':Jopt-coeff['J_population']})
                coefficients.append(coeff);pop[(sd,lam)]=coeff
            for n in CFG['finite_sample']['sample_sizes']:
                count=rng.multinomial(n,np.ones(3)/3,size=reps)
                wins=rng.binomial(count,truep[None,:])
                gap_sum=count*gap[None,:]+rng.normal(size=(reps,3))*sd*np.sqrt(count)
                values={}
                for lam in CFG['finite_sample']['rating_weights']:
                    lo=np.full(reps,lo0);hi=np.full(reps,hi0)
                    def derivative(t):
                        return np.sum(d[None,:]*(count*(expit(t[:,None]*d[None,:])+lam*t[:,None]*d[None,:])-wins-lam*gap_sum),axis=1)/n
                    left_derivative=derivative(lo);right_derivative=derivative(hi)
                    for _ in range(55):
                        mid=(lo+hi)/2;g=derivative(mid)
                        lo=np.where(g<0,mid,lo);hi=np.where(g>=0,mid,hi)
                    theta=(lo+hi)/2
                    val=scalar_values(theta,f,r,ref);values[lam]=val
                    difference=val-values[0.]
                    mean=float(np.mean(difference));se=float(np.std(difference,ddof=1)/np.sqrt(reps))
                    coeff=pop[(sd,lam)];base=pop[(sd,0.)]
                    expectation=coeff['J_population']-base['J_population']-(coeff['expected_regret_1_over_n_coefficient']-base['expected_regret_1_over_n_coefficient'])/n
                    rows.append({'case':name,'n':n,'rating_noise_sd':sd,'lambda':lam,
                                 'replicates':reps,'mean_value':float(np.mean(val)),
                                 'mean_restricted_regret':float(Jopt-np.mean(val)),
                                 'mean_value_minus_binary':mean,'paired_MC_standard_error':se,
                                 'paired_MC_95_lower':mean-1.96*se,'paired_MC_95_upper':mean+1.96*se,
                                 'theta_mean':float(np.mean(theta)),
                                 'theta_bias_times_n':float(n*(np.mean(theta)-coeff['theta_population'])),
                                 'boundary_fraction':float(np.mean((left_derivative>=0)|(right_derivative<=0))),
                                 'population_value_minus_binary':coeff['J_population']-base['J_population'],
                                 'asymptotic_expected_difference':expectation})
        print(f'Finite-sample case {name} complete',flush=True)
    write_csv(OUT/'finite_sample.csv',rows);write_csv(OUT/'asymptotic_coefficients.csv',coefficients)
    return rows,coefficients


def summarize(pop,diagnostics,samples,coefficients):
    harmful={lam:sum(x['value_minus_binary']<-1e-6 for x in pop if x['lambda']==lam and x['reward_mode'] in ('random','nonmonotone') and x['rank']<x['actions']-1)
             for lam in CFG['population']['rating_weights'][1:]}
    eligible=sum(x['lambda']==0 and x['reward_mode'] in ('random','nonmonotone') and x['rank']<x['actions']-1 for x in pop)
    appreciable=[x for x in diagnostics if abs(x['sensitivity'])>1e-5]
    summary={'scope':CFG['scope'],'population_cases':len(diagnostics),'population_fits':len(pop),
             'restricted_random_or_nonmonotone_cases':eligible,'harmful_count_by_lambda':harmful,
             'full_rank_cases':sum(x['lambda']==0 and x['rank']==x['actions']-1 for x in pop),
             'declared_realizable_cases':sum(x['lambda']==0 and x['reward_mode']=='realizable' for x in pop),
             'local_sign_cases':len(appreciable),'local_sign_matched':sum(x['sign_matched'] for x in appreciable),
             'population_max_gradient_inf':max(x['gradient_inf'] for x in pop),
             'finite_sample_rows':len(samples),'replicates_per_row':CFG['finite_sample']['replicates'],
             'max_boundary_fraction':max(x['boundary_fraction'] for x in samples),
             'limitations':['Synthetic instance distribution is chosen, not an estimate of LLM prevalence.',
                            'Monte Carlo intervals are descriptive pointwise intervals, not multiplicity-adjusted discovery tests.',
                            'Finite-sample cases and noise levels are controlled synthetic mechanisms, not human feedback.',
                            'No language-task, baseline training cost or adapter experiment is included.']}
    return summary


def figures(pop,samples):
    fig,ax=plt.subplots(figsize=(7.0,4.4))
    vals=[x for x in pop if x['case']=='original_negative']
    ax.plot([x['rating_gap_mse'] for x in vals],[x['value'] for x in vals],'o-')
    for x in vals:
        offset=(-48,16) if x['lambda']==0 else (-58,-19) if x['lambda']==.001 else (4,5)
        ax.annotate(f"λ={x['lambda']:g}",(x['rating_gap_mse'],x['value']),xytext=offset,textcoords='offset points',fontsize=8)
    ax.set_ylim(4.15,5.48)
    ax.set(xlabel='Exact reward-gap MSE',ylabel='KL-regularized policy value',title='Constructed population example: better fit, worse decisions')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'population_fit_value.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=False)
    for ax,name in zip(axes,['rescaled_negative','realizable_saturated']):
        for lam in [.01,.1,1.]:
            rows=[x for x in samples if x['case']==name and x['rating_noise_sd']==0. and x['lambda']==lam]
            n=np.array([x['n'] for x in rows]);y=np.array([x['mean_value_minus_binary'] for x in rows]);err=np.array([1.96*x['paired_MC_standard_error'] for x in rows])
            ax.errorbar(n,y,yerr=err,marker='o',label=f'λ={lam:g}',capsize=2)
        ax.axhline(0,color='black',linewidth=.8);ax.set_xscale('log',base=2)
        ax.set(xlabel='Number of observed pairs',ylabel='Value difference from binary-only',title=name.replace('_',' '))
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Shared-data Monte Carlo; exact gaps; pointwise 95% MC intervals')
    fig.tight_layout();fig.savefig(OUT/'sample_budget_transfer.png',dpi=180);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True);start=time.time()
    manifest={'config_sha256':hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'python':platform.python_version(),'numpy':np.__version__,
              'seed':CFG['seed'],'started_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'revision_note':'Canonical outputs only: result tables, PNG figures and this manifest; settings and random draws unchanged.',
              'status':'running'}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rng=np.random.default_rng(CFG['seed'])
    pop,diagnostics=population_study(rng)
    samples,coefficients=sample_study(rng)
    summary=summarize(pop,diagnostics,samples,coefficients);figures(pop,samples)
    manifest.update({'status':'completed','elapsed_seconds':time.time()-start,'summary':summary})
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
