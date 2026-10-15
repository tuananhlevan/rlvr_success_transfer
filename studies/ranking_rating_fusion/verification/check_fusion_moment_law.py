"""Independent high-precision finite-K checks of the fourth/fifth-order law.
Uses Fraction coefficients and Decimal population roots/value: no MC, no ML.
"""
import json
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path

HERE=Path(__file__).resolve().parent


def mean(x,p):return sum(a*b for a,b in zip(x,p))


def constants(f,h,ref,matched):
    edges=[(i,j) for i in range(len(f)) for j in range(i+1,len(f))]
    weights=[ref[i]*ref[j] if matched else F(1) for i,j in edges]
    q=[w/sum(weights) for w in weights]
    d=[f[i]-f[j] for i,j in edges];g=[h[i]-h[j] for i,j in edges]
    mu2=mean([x*x for x in d],q);mu4=mean([x**4 for x in d],q)
    k=mean([x*y for x,y in zip(d,g)],q)/mu2
    b3=(mu4*k**3-mean([x*y**3 for x,y in zip(d,g)],q))/(12*mu2)
    mf,mh=mean(f,ref),mean(h,ref)
    cf=[x-mf for x in f];ch=[x-mh for x in h]
    var=mean([x*x for x in cf],ref);cov=mean([x*y for x,y in zip(cf,ch)],ref)
    c1=cov-k*var
    c2=k*mean([x*x*y for x,y in zip(cf,ch)],ref)-k*k*mean([x**3 for x in cf],ref)
    order=5 if c1==0 else 4
    lam=F(1,10);P=(c2 if order==5 else c1)*b3*4*lam/(1+4*lam)
    return edges,q,{'k':k,'b3':b3,'c1':c1,'c2':c2,'P':P,'order':order}


def D(x):return Decimal(x.numerator)/Decimal(x.denominator)


def population_roots(f,h,ref,edges,q,eps,lam):
    f,h,ref,q=[list(map(D,x)) for x in (f,h,ref,q)]
    r=[eps*x for x in h]
    def sigmoid(x):return 1/(1+(-x).exp())
    def score(t):
        return sum(w*(f[i]-f[j])*(sigmoid(t*(f[i]-f[j]))-sigmoid(r[i]-r[j])
                       +lam*(t*(f[i]-f[j])-(r[i]-r[j]))) for w,(i,j) in zip(q,edges))
    lo,hi=Decimal(-2),Decimal(2)
    assert score(lo)<0<score(hi)
    for _ in range(180):
        mid=(lo+hi)/2
        if score(mid)<0:lo=mid
        else:hi=mid
    t=(lo+hi)/2
    raw=[p*(t*x).exp() for p,x in zip(ref,f)];total=sum(raw);p=[x/total for x in raw]
    value=sum(a*b for a,b in zip(p,r))-sum(a*(a/b).ln() for a,b in zip(p,ref))
    return t,value


def main():
    cases=[('three_matched',[F(0),F(7,10),F(1)],[F(0),F(1),F(2,5)],[F(1,3)]*3,True),
           ('four_matched',[F(0),F(1,5),F(4,5),F(7,5)],[F(3,10),F(1),F(2,5),F(-1,5)],
            [F(1,10),F(1,5),F(3,10),F(2,5)],True),
           ('four_unmatched',[F(0),F(1,5),F(4,5),F(7,5)],[F(3,10),F(1),F(2,5),F(-1,5)],
            [F(1,10),F(1,5),F(3,10),F(2,5)],False)]
    rows=[];exact=[]
    with localcontext() as ctx:
        ctx.prec=70
        for name,f,h,ref,matched in cases:
            edges,q,c=constants(f,h,ref,matched)
            assert c['P']!=0
            exact.append({'case':name,'coefficients':{key:str(v) for key,v in c.items()},
                          'features':[str(v) for v in f],'reward_direction':[str(v) for v in h],
                          'reference':[str(v) for v in ref],'pair_design':'reference' if matched else 'uniform'})
            errors=[]
            for ep in [F(1,5),F(1,10),F(1,20),F(1,50),F(1,100)]:
                eps=D(ep)
                tb,jb=population_roots(f,h,ref,edges,q,eps,Decimal(0))
                tl,jl=population_roots(f,h,ref,edges,q,eps,Decimal('0.1'))
                observed=(jb-jl)/eps**c['order'];predicted=D(c['P'])
                rel=abs((observed-predicted)/predicted);errors.append(rel)
                rows.append({'case':name,'epsilon':float(ep),'leading_order':c['order'],
                             'P_exact':str(c['P']),'P_decimal':str(predicted),
                             'observed_regret_difference_scaled':str(observed),'relative_error':str(rel),
                             'theta_binary':str(tb),'theta_fusion':str(tl)})
            assert errors[-1]<Decimal('.05')
            assert errors[-1]<errors[0]
    out={'scope':'high-precision finite-action coefficients and roots, not a statistical or neural experiment',
         'decimal_precision':70,'cases':exact,'population_checks':len(rows),'checks':rows,
         'assertions_passed':True}
    (HERE/'fusion_moment_law_checks_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'population_checks':len(rows),'assertions_passed':True},indent=2))


if __name__=='__main__':main()
