from __future__ import annotations
from pathlib import Path
import csv,json
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[2]
FIG=OUT/'figures';TAB=OUT/'tables';FIG.mkdir(parents=True,exist_ok=True);TAB.mkdir(parents=True,exist_ok=True)
BLUE='#21618C';ORANGE='#C66A2C';INK='#20252B';MUTED='#65717E'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.4,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
PREF={'eta_gain':.2,'eta_loss':.05,'alpha':.88,'loss_aversion':2.25,'probability_epsilon':.05,'value_epsilon':.01,'beta_gain':.61,'beta_loss':.69,'cost':.001,'trade_fraction':.2,'initial_risky_weight':.5,'terminal_wealth':1.015,'future_risky_returns':[-.04,.04],'candidate_target_risky_weights':[.2,.8]}
PATHS={'Rise then fall':[1.,1.10,1.015],'Fall then rise':[1.,.90,1.015]}
RULES={'Asymmetric':(.2,.05),'Symmetric':(.125,.125),'Static':(0.,0.)}

def references(path,rates):
    r=1.;out=[r]
    for x in path[1:]: r += (rates[0] if x>=r else rates[1])*(x-r);out.append(r)
    return np.asarray(out)

def weight(p,beta):
    e=PREF['probability_epsilon']
    def tk(q): return q**beta/(q**beta+(1-q)**beta)**(1/beta)
    p=np.asarray(p);return (tk(e+(1-2*e)*p)-tk(e))/(tk(1-e)-tk(e))

def exact_cpt(reference,p,target_risky,rates):
    X=PREF['terminal_wealth'];chi=PREF['trade_fraction']; risky=(1-chi)*.5+chi*target_risky
    turnover=2*abs(risky-.5); terminal=X*(1+risky*np.asarray(PREF['future_risky_returns'])-PREF['cost']*turnover)
    rnew=reference+np.where(terminal>=reference,rates[0],rates[1])*(terminal-reference);y=terminal-rnew
    probs=np.asarray([1-p,p]);total=0.
    for sign,beta,mult in [(1,PREF['beta_gain'],1.),(-1,PREF['beta_loss'],PREF['loss_aversion'])]:
        mag=np.maximum(sign*y,0.);v=mult*((mag**2+PREF['value_epsilon']**2)**(PREF['alpha']/2)-PREF['value_epsilon']**PREF['alpha'])
        order=np.argsort(v,kind='stable');v=v[order];pr=probs[order];surv=1-np.r_[0.,np.cumsum(pr)[:-1]]
        total += sign*np.dot(np.diff(np.r_[0.,v]),weight(surv,beta))
    return float(total)

def save(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(FIG/f'{name}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

def main():
    ref_rows=[]
    for rule,rates in RULES.items():
        for hist,path in PATHS.items():
            r=references(path,rates);ref_rows.append(dict(rule=rule,history=hist,terminal_wealth=path[-1],terminal_reference=r[-1],relative_outcome=path[-1]-r[-1]))
    pdict={(r['rule'],r['history']):r for r in ref_rows}
    pgrid=np.linspace(.05,.95,901);curves=[];thresholds={}
    for hist,path in PATHS.items():
        r=references(path,RULES['Asymmetric'])[-1]
        lo=np.array([exact_cpt(r,p,.2,RULES['Asymmetric']) for p in pgrid]);hi=np.array([exact_cpt(r,p,.8,RULES['Asymmetric']) for p in pgrid])
        fn=lambda p:exact_cpt(r,p,.8,RULES['Asymmetric'])-exact_cpt(r,p,.2,RULES['Asymmetric'])
        thresholds[hist]=float(brentq(fn,.05,.95))
        for p,a,b in zip(pgrid,lo,hi):curves.append(dict(history=hist,p_up=float(p),J_risky_44=float(a),J_risky_56=float(b)))
    with (TAB/'e1_reference_values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ref_rows[0]));w.writeheader();w.writerows(ref_rows)
    with (TAB/'e1_exact_choice_curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    summary=[]
    for hist in PATHS:
        r=pdict[('Asymmetric',hist)];j44=exact_cpt(r['terminal_reference'],.7,.2,RULES['Asymmetric']);j56=exact_cpt(r['terminal_reference'],.7,.8,RULES['Asymmetric'])
        summary.append(dict(history=hist,terminal_wealth=r['terminal_wealth'],terminal_reference=r['terminal_reference'],relative_outcome=r['relative_outcome'],domain='gain' if r['relative_outcome']>=0 else 'loss',J_44_p07=j44,J_56_p07=j56,preferred_risky_share='56%' if j56>j44 else '44%',switch_probability=thresholds[hist]))
    with (TAB/'e1_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (TAB/'e1_summary.json').write_text(json.dumps({'preference':PREF,'summary':summary},indent=2),encoding='utf-8')

    fig,axs=plt.subplots(2,2,figsize=(11.4,7.8));
    ax=axs[0,0]
    for (hist,path),c in zip(PATHS.items(),[BLUE,ORANGE]):
        r=references(path,RULES['Asymmetric']);x=np.arange(3)
        ax.plot(x,path,'o-',lw=2,color=c,label=f'{hist}: wealth');ax.plot(x,r,'s--',lw=1.6,ms=4,color=c,alpha=.72,label=f'{hist}: reference')
    ax.set_xticks([0,1,2],['Start','History','Matched endpoint']);ax.set_ylabel('Normalized wealth / reference');ax.set_title('A  History separates the reference point',loc='left',fontweight='bold');ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,fontsize=7.4,ncol=2)

    ax=axs[0,1];rules=list(RULES);xx=np.arange(len(rules));w=.34
    vals1=[pdict[(r,'Rise then fall')]['relative_outcome'] for r in rules];vals2=[pdict[(r,'Fall then rise')]['relative_outcome'] for r in rules]
    ax.bar(xx-w/2,vals1,w,label='Rise then fall',color=BLUE);ax.bar(xx+w/2,vals2,w,label='Fall then rise',color=ORANGE);ax.axhline(0,color=INK,lw=.9)
    ax.set_xticks(xx,rules);ax.set_ylabel(r'Terminal relative outcome $X_T-r_T$');ax.set_title('B  The same wealth enters different CPT domains',loc='left',fontweight='bold');ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,fontsize=8)

    for ax,(hist,c,label) in zip([axs[1,0],axs[1,1]],[("Rise then fall",BLUE,'C'),("Fall then rise",ORANGE,'D')]):
        r=pdict[('Asymmetric',hist)]['terminal_reference']; j44=np.array([exact_cpt(r,p,.2,RULES['Asymmetric']) for p in pgrid]);j56=np.array([exact_cpt(r,p,.8,RULES['Asymmetric']) for p in pgrid])
        ax.plot(pgrid,j44,lw=2,color='#526D82',label='44% risky');ax.plot(pgrid,j56,lw=2,color=c,label='56% risky');ax.axvline(.7,color=INK,ls=':',lw=1.1)
        y44=exact_cpt(r,.7,.2,RULES['Asymmetric']);y56=exact_cpt(r,.7,.8,RULES['Asymmetric']);ax.scatter([.7,.7],[y44,y56],s=23,color=['#526D82',c],zorder=3)
        pref='44% risky' if y44>y56 else '56% risky';ax.annotate(f'$p=0.7$: {pref} preferred',xy=(.7,max(y44,y56)),xytext=(.44,.88),textcoords='axes fraction',fontsize=8.6,arrowprops=dict(arrowstyle='-',color=MUTED),color=INK)
        ax.set_xlabel('Probability of +4% risky return, $p$');ax.set_ylabel('Exact CPT objective');ax.set_title(f'{label}  {hist}',loc='left',fontweight='bold');ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();save(fig,'e1_path_and_decision_final')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
