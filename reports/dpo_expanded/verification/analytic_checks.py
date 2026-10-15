"""Independent finite-action checks; no training, network or third-party packages."""
import json, math
from pathlib import Path
F = (-1., 0., 1.)

def sig(x):
    return 1/(1+math.exp(-x))
def prob(theta):
    z = sum(math.exp(theta*f) for f in F)
    return [math.exp(theta*f)/z for f in F]
def bisect(fn, lo, hi, iterations=90):
    assert fn(lo)*fn(hi) <= 0
    for _ in range(iterations):
        mid=(lo+hi)/2
        if fn(lo)*fn(mid)<=0: hi=mid
        else: lo=mid
    return (lo+hi)/2

def direct_gradient(theta,u,a,gamma):
    p=prob(theta); r=(0.,a,0.)
    return sum(p[i]*p[j]*gamma*(F[i]-F[j])*
       (sig(gamma*u*(F[i]-F[j]))-sig(r[i]-r[j]))
       for i in range(3) for j in range(3))
def closed_gradient(theta,u,a,gamma):
    z=1+2*math.cosh(theta); b=math.tanh(a/2)
    return 2*gamma/z**2*(math.cosh(theta)*math.tanh(gamma*u/2)
       +math.tanh(gamma*u)+b*math.sinh(theta))
def w(theta,a):
    if theta==0: return 0.
    return bisect(lambda x: math.cosh(theta)*math.tanh(x/2)+math.tanh(x)
                 -math.tanh(a/2)*math.sinh(theta),0,a)
def wp(theta,a):
    x=w(theta,a); b=math.tanh(a/2)
    return (b/math.cosh(theta)+math.tanh(theta)*math.tanh(x))/(
      .5*math.cosh(theta)/math.cosh(x/2)**2+1/math.cosh(x)**2)
def objective(theta,a,gamma):
    p=prob(theta)
    return a*p[1]-gamma*sum(v*math.log(3*v) for v in p)

a=2.; gamma=.55; R=3.
errs=[abs(direct_gradient(t,u,a,g)-closed_gradient(t,u,a,g))
      for t in (-2.,-.7,0.,.3,2.1) for u in (-2.2,0.,.4,1.7)
      for g in (.3,.55,1.)]
assert max(errs)<1e-14
cycles=[]
for lo,hi in ((1.,1.5),(2.5,3.)):
    root=bisect(lambda t:w(t,a)/gamma-t,lo,hi)
    cycles.append({'magnitude':root,'one_step_abs_derivative':wp(root,a)/gamma,
       'two_step_multiplier':(wp(root,a)/gamma)**2,
       'true_regularized_regret':objective(0,a,gamma)-objective(root,a,gamma)})
assert cycles[0]['two_step_multiplier']>1 and cycles[1]['two_step_multiplier']<1
phase=[(t,w(t,a)/t) for t in [i*R/600 for i in range(1,601)]]
maximum=max(phase,key=lambda row:row[1])
local=2*math.tanh(a/2)/3
assert maximum[1]>local

# Inner-depth formula checked against the actual independent ordered-pair gradient.
g=.3; eta=1.; H=g*g/3; c=2*math.tanh(a/2)/(3*g)
def tq(theta,q):
    u=theta
    for _ in range(q): u-=eta*direct_gradient(theta,u,a,g)
    return u
depth=[]
for q in (1,5,15,30,50):
    numerical=(tq(1e-5,q)-tq(-1e-5,q))/2e-5
    formula=(1+c)*(1-eta*H)**q-c
    assert abs(numerical-formula)<1e-8
    depth.append({'steps':q,'finite_difference':numerical,'formula':formula})

# Independent reward-optimization endpoint construction.
t=1.; delta=.2; p=prob(t); m=sum(v*f for v,f in zip(p,F))
h=(1.,0.,p[0]*(1+m)/(p[2]*(1-m)))
def cov(theta,h):
    p=prob(theta); mf=sum(v*f for v,f in zip(p,F)); mh=sum(v*z for v,z in zip(p,h))
    return sum(v*f*z for v,f,z in zip(p,F,h))-mf*mh
def var(theta):
    p=prob(theta); mf=sum(v*f for v,f in zip(p,F))
    return sum(v*f*f for v,f in zip(p,F))-mf*mf
def k(theta): return cov(theta,h)/var(theta)
def util(theta,linear,offset):
    p=prob(theta)
    return sum(v*(linear*f+offset*z) for v,f,z in zip(p,F,h)) - sum(v*math.log(3*v) for v in p)
assert abs(cov(t,h))<1e-15
endpoint=bisect(lambda x:t-x+delta*k(x),-3,3)
mixed=bisect(lambda x:-x+delta*k(x)/2,-3,3)
assert abs(endpoint-1)<1e-12 and abs(mixed)>1e-3
endpoint_grid=max((util(x/1000,t,delta),x/1000) for x in range(-3000,3001))
assert endpoint_grid[1]==1.
regret=util(mixed,0,delta/2)-util(0,0,delta/2)

# Two unknown-link utilities preserve binary probabilities but differ in deployment.
def gibbs(u):
    z=sum(math.exp(v) for v in u); return [math.exp(v)/z for v in u]
pu=gibbs([0,1,3]); pv=gibbs([0,1,4]); bc=sum(math.sqrt(x*y) for x,y in zip(pu,pv))
floor=-math.log(bc)
assert floor>0
# A fixed externally supplied target z=.7 makes the printed CATTO affine loss
# favor confidence 1 rather than calibrated confidence .7.
def catto(c,z): return z*(1-c)+(1-z)*c
assert catto(1,.7)<catto(.7,.7)
# Printed CITA offline anchor is unbounded below with unchanged DPO margin.
cita=[{'epsilon':e,'dpo_loss':math.log(2),'offline_anchor':math.log(3*e)}
      for e in (.1,.01,.001,1e-6)]
# Common token-row permutations leave the complete Gram matrix unchanged.
X=[[1,2],[3,0],[-2,4]]
def gram(x): return [[sum(row[i]*row[j] for row in x) for j in range(2)] for i in range(2)]
assert gram(X)==gram([X[2],X[0],X[1]])
# A sensitivity mixture has nonadditive logits despite common preference ordering.
def mixlogit(x):
    p=.5*sig(.5*x)+.5*sig(2*x); return math.log(p/(1-p))
cycle_defect=mixlogit(2)-2*mixlogit(1)
assert abs(cycle_defect)>1e-3
out={'scope':'deterministic finite-action analytical checks; not LLM experiments',
 'maximum_direct_gradient_error':max(errs),'feedback':{
 'a':a,'gamma':gamma,'R':R,'local_threshold':local,
 'global_threshold_grid_lower_bound':maximum[1], 'grid_argmax':maximum[0],
 'cycles':cycles,'inner_depth':depth},
 'endpoint_compression':{'h':h,'component_optimum':endpoint,'mixed_optimum':mixed,
 'midpoint_regret':regret,'grid_component_optimum':endpoint_grid[1]},
 'unknown_link':{'policy_u':pu,'policy_v':pv,'regret_lower_bound':floor},
 'formula_audits':{'mix_logit_cycle_defect':cycle_defect,'catto_loss_at_target':catto(.7,.7),
 'catto_loss_at_boundary':catto(1,.7),'cita_unbounded_sequence':cita,
 'spinal_gram_invariant_to_row_permutation':True},'all_assertions_passed':True}
Path(__file__).with_name('analytic_checks_results.json').write_text(json.dumps(out,indent=2)+'\n')
# Standalone scientific figure without package dependencies.
W,H=720,380
pts=' '.join(f'{60+600*t/R:.2f},{320-250*val:.2f}' for t,val in phase)
ly=320-250*local; gy=320-250*gamma
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="100%" height="100%" fill="white"/>
<g font-family="sans-serif" font-size="14" fill="#222">
<text x="60" y="25">Exact population feedback: global and local thresholds differ (a = 2)</text>
<path d="M60 65 V320 H670" fill="none" stroke="#333"/>
<polyline points="{pts}" fill="none" stroke="#185abd" stroke-width="2.5"/>
<path d="M60 {ly} H660" stroke="#777" stroke-dasharray="5 4"/>
<path d="M60 {gy} H660" stroke="#b33" stroke-dasharray="5 4"/>
<text x="430" y="145" fill="#185abd">w(theta) / theta</text>
<text x="70" y="{ly+19}">local threshold = {local:.4f}</text>
<text x="70" y="{gy-9}" fill="#b33">gamma = 0.55: bistable example</text>
<text x="340" y="355">parameter magnitude theta</text>
<text x="55" y="338">0</text><text x="650" y="338">3</text>
</g></svg>'''
Path(__file__).with_name('feedback_phase.svg').write_text(svg)
print(json.dumps(out,indent=2))
