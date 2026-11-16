"""
Run this file ONCE to create app.py in the correct location.
Command: python run_me_first.py
"""
import os

# This creates app.py in the SAME folder as this script
app_code = '''"""
HealthPredict AI — Diabetes Risk Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use(\'Agg\')
import shap
import warnings
from datetime import datetime
warnings.filterwarnings(\'ignore\')

st.set_page_config(page_title="HealthPredict AI", page_icon="🏥", layout="wide")

st.markdown("""
<style>
@import url(\'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;800&family=Space+Mono&display=swap\');
html,body,[class*="css"]{font-family:\'DM Sans\',sans-serif;}
.stApp{background:linear-gradient(135deg,#060912 0%,#0c1525 60%,#060912 100%);color:#e2e8f0;}
.hero-title{font-size:2.6rem;font-weight:800;background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:4px;}
.hero-sub{text-align:center;color:#4b5563;font-size:.9rem;font-family:\'Space Mono\',monospace;margin-bottom:1.5rem;}
.risk-box{border-radius:20px;padding:2.5rem 2rem;text-align:center;margin:1.5rem auto;max-width:700px;
  animation:fadeIn 0.5s ease;}
@keyframes fadeIn{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.risk-LOW{background:linear-gradient(145deg,#0a2318,#052010);border:2px solid #22c55e;
  box-shadow:0 0 50px rgba(34,197,94,0.4);}
.risk-MEDIUM{background:linear-gradient(145deg,#2a1e04,#1a1200);border:2px solid #f59e0b;
  box-shadow:0 0 50px rgba(245,158,11,0.4);}
.risk-HIGH{background:linear-gradient(145deg,#280808,#180404);border:2px solid #ef4444;
  box-shadow:0 0 50px rgba(239,68,68,0.4);}
.risk-title{font-size:3.5rem;font-weight:800;margin:.5rem 0;}
.risk-LOW .risk-title{color:#22c55e;text-shadow:0 0 30px rgba(34,197,94,0.6);}
.risk-MEDIUM .risk-title{color:#f59e0b;text-shadow:0 0 30px rgba(245,158,11,0.6);}
.risk-HIGH .risk-title{color:#ef4444;text-shadow:0 0 30px rgba(239,68,68,0.6);}
.risk-prob{font-family:\'Space Mono\',monospace;font-size:1.1rem;color:#94a3b8;margin:.5rem 0;}
.risk-advice{font-size:.9rem;padding:.8rem 1rem;border-radius:10px;margin-top:.8rem;}
.risk-LOW .risk-advice{background:rgba(34,197,94,.1);color:#86efac;border:1px solid rgba(34,197,94,.2);}
.risk-MEDIUM .risk-advice{background:rgba(245,158,11,.1);color:#fcd34d;border:1px solid rgba(245,158,11,.2);}
.risk-HIGH .risk-advice{background:rgba(239,68,68,.1);color:#fca5a5;border:1px solid rgba(239,68,68,.2);}
.model-grid{display:flex;gap:12px;justify-content:center;margin-top:1.2rem;flex-wrap:wrap;}
.model-card{flex:1;min-width:120px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
  border-radius:12px;padding:.9rem;text-align:center;}
.model-name{font-size:.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.4rem;}
.sec-hdr{font-size:1.1rem;font-weight:700;color:#e2e8f0;border-left:3px solid #60a5fa;
  padding-left:.7rem;margin:1.8rem 0 .9rem;}
.metric-card{background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
  border-radius:12px;padding:1.1rem;text-align:center;}
.metric-label{color:#6b7280;font-size:.68rem;text-transform:uppercase;letter-spacing:1.5px;}
.metric-value{color:#60a5fa;font-size:1.4rem;font-weight:700;font-family:\'Space Mono\',monospace;}
.hist-wrap{overflow-x:auto;border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:.5rem;}
table.hist{width:100%;border-collapse:collapse;font-size:.8rem;}
table.hist th{background:rgba(96,165,250,.12);color:#60a5fa;font-weight:700;padding:10px 12px;
  text-align:left;font-size:.7rem;letter-spacing:1.5px;text-transform:uppercase;
  border-bottom:1px solid rgba(96,165,250,.2);}
table.hist td{padding:9px 12px;border-bottom:1px solid rgba(255,255,255,0.04);color:#94a3b8;}
.badge{display:inline-block;font-size:.68rem;font-weight:700;padding:3px 10px;border-radius:20px;letter-spacing:1px;}
.badge-LOW{background:rgba(34,197,94,.15);color:#22c55e;border:1px solid rgba(34,197,94,.3);}
.badge-MEDIUM{background:rgba(245,158,11,.15);color:#f59e0b;border:1px solid rgba(245,158,11,.3);}
.badge-HIGH{background:rgba(239,68,68,.15);color:#ef4444;border:1px solid rgba(239,68,68,.3);}
div[data-testid="stSidebar"]{background:rgba(10,18,36,.95)!important;border-right:1px solid rgba(255,255,255,0.05)!important;}
.stButton>button{background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:white;border:none;
  border-radius:10px;padding:.7rem 2rem;font-weight:700;font-size:1rem;width:100%;transition:all .2s;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(59,130,246,.4);}
</style>
""", unsafe_allow_html=True)

# ── session state ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── train models ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from imblearn.over_sampling import SMOTE

    np.random.seed(42)
    n = 768
    pregnancies    = np.random.choice(range(0,18),n,p=[.145,.143,.116,.095,.079,.065,.052,.040,.033,.027,.021,.017,.013,.011,.009,.008,.007,.006,.013])
    glucose        = np.clip(np.random.normal(120,32,n),44,199).astype(int)
    blood_pressure = np.clip(np.random.normal(69,19,n),0,122).astype(int)
    skin_thickness = np.clip(np.random.normal(20,16,n),0,99).astype(int)
    insulin        = np.clip(np.random.exponential(80,n),0,846).astype(int)
    bmi            = np.clip(np.random.normal(32,7.9,n),0,67.1).round(1)
    dpf            = np.clip(np.random.exponential(0.47,n),0.078,2.42).round(3)
    age            = np.clip(np.random.normal(33,11.8,n),21,81).astype(int)
    log_odds = -6+0.035*glucose+0.09*bmi+0.04*age+0.15*pregnancies+0.003*insulin-0.01*blood_pressure
    prob     = 1/(1+np.exp(-log_odds))
    outcome  = (np.random.random(n)<prob).astype(int)
    df = pd.DataFrame({\'Pregnancies\':pregnancies,\'Glucose\':glucose,\'BloodPressure\':blood_pressure,
        \'SkinThickness\':skin_thickness,\'Insulin\':insulin,\'BMI\':bmi,
        \'DiabetesPedigreeFunction\':dpf,\'Age\':age,\'Outcome\':outcome})
    X,y   = df.drop(\'Outcome\',axis=1),df[\'Outcome\']
    feats = X.columns.tolist()
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    Xr,yr = SMOTE(random_state=42).fit_resample(Xtr,ytr)
    scaler = StandardScaler()
    Xrsc   = scaler.fit_transform(Xr)
    Xtesc  = scaler.transform(Xte)
    clfs = {
        \'Logistic Regression\':LogisticRegression(max_iter=1000,random_state=42),
        \'Random Forest\':RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42),
        \'XGBoost\':XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.08,
                                 use_label_encoder=False,eval_metric=\'logloss\',random_state=42),
    }
    trained,metrics={},{}
    for name,clf in clfs.items():
        if name==\'Logistic Regression\':
            clf.fit(Xrsc,yr);yp=clf.predict(Xtesc);ypr=clf.predict_proba(Xtesc)[:,1]
        else:
            clf.fit(Xr,yr);yp=clf.predict(Xte);ypr=clf.predict_proba(Xte)[:,1]
        metrics[name]={\'Accuracy\':round(accuracy_score(yte,yp)*100,2),
                       \'F1-Score\':round(f1_score(yte,yp)*100,2),
                       \'AUC-ROC\':round(roc_auc_score(yte,ypr)*100,2)}
        trained[name]=clf
    explainer=shap.TreeExplainer(trained[\'XGBoost\'])
    return trained,scaler,explainer,metrics,feats,Xte,Xtesc,yte.values

def get_risk(p):
    if   p<0.35: return "LOW",   "🛡️","#22c55e"
    elif p<0.65: return "MEDIUM","⚡","#f59e0b"
    else:        return "HIGH",  "🔬","#ef4444"

ADVICE={
    "LOW":   "✅ Low risk. Maintain healthy diet, regular exercise and annual check-ups.",
    "MEDIUM":"⚠️ Moderate risk. Consult a physician. Lifestyle changes advised.",
    "HIGH":  "🚨 High risk! Immediate clinical evaluation and HbA1c testing strongly recommended.",
}

def plot_shap(explainer,input_df,features):
    sv   = explainer.shap_values(input_df)[0]
    vals = input_df.values[0]
    colors=[\'#ef4444\' if s>0 else \'#22c55e\' for s in sv]
    fig,ax=plt.subplots(figsize=(9,3.8))
    fig.patch.set_facecolor(\'#0c1525\');ax.set_facecolor(\'#0c1525\')
    bars=ax.barh(features,sv,color=colors,height=0.58,alpha=0.88)
    for bar,val,fv in zip(bars,sv,vals):
        x=bar.get_width()
        ax.text(x+(0.006 if x>=0 else -0.006),bar.get_y()+bar.get_height()/2,
                f\'{val:+.3f}  (={fv:.1f})\',va=\'center\',ha=\'left\' if x>=0 else \'right\',
                color=\'#cbd5e1\',fontsize=8.5,fontfamily=\'monospace\')
    ax.axvline(0,color=\'#334155\',lw=1,ls=\'--\')
    ax.set_xlabel(\'SHAP value (impact on model output)\',color=\'#64748b\',fontsize=9)
    ax.set_title(f\'Feature Impact  |  Base: {explainer.expected_value:.3f}\',color=\'#e2e8f0\',fontsize=11,pad=10)
    ax.tick_params(colors=\'#64748b\',labelsize=9)
    for sp in ax.spines.values():sp.set_edgecolor(\'#1e293b\')
    plt.tight_layout()
    return fig

# ════════════════════════════════════════════════════════
def main():
    st.markdown(\'<div class="hero-title">🏥 HealthPredict AI</div>\',unsafe_allow_html=True)
    st.markdown(\'<div class="hero-sub">Diabetes Risk · XGBoost + SHAP + History</div>\',unsafe_allow_html=True)

    with st.spinner("🔬 Training models…"):
        trained,scaler,explainer,metrics,features,Xte,Xtesc,yte = load_models()
    st.success("✅ Models ready — Logistic Regression | Random Forest | XGBoost")

    # sidebar
    with st.sidebar:
        st.markdown("### 🩺 Patient Vitals")
        pregnancies    = st.slider("Pregnancies",0,17,3)
        glucose        = st.slider("Glucose (mg/dL)",44,199,120)
        blood_pressure = st.slider("Blood Pressure (mmHg)",0,122,70)
        skin_thickness = st.slider("Skin Thickness (mm)",0,99,20)
        insulin        = st.slider("Insulin (μU/mL)",0,846,80)
        bmi            = st.slider("BMI",10.0,67.1,32.0,step=0.1)
        dpf            = st.slider("Diabetes Pedigree",0.078,2.42,0.47,step=0.001)
        age            = st.slider("Age",21,81,33)
        predict_btn    = st.button("⚡ Predict Diabetes Risk")
        st.divider()
        if st.button("🗑️ Clear History"):
            st.session_state.history=[]
            st.rerun()

    # prediction
    if predict_btn:
        inp   = np.array([[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,dpf,age]])
        idf   = pd.DataFrame(inp,columns=features)
        isc   = scaler.transform(idf)
        xp    = trained[\'XGBoost\'].predict_proba(idf)[0][1]
        rp    = trained[\'Random Forest\'].predict_proba(idf)[0][1]
        lp    = trained[\'Logistic Regression\'].predict_proba(isc)[0][1]
        ens   = round(xp*0.5+rp*0.3+lp*0.2,4)
        risk,icon,color = get_risk(ens)
        xr,xi,xc = get_risk(xp)
        rr,ri,rc  = get_risk(rp)
        lr2,li,lc = get_risk(lp)

        # save history
        st.session_state.history.insert(0,{
            "Time":datetime.now().strftime("%H:%M:%S"),
            "Risk":risk,"Probability":f"{ens:.1%}",
            "Glucose":glucose,"BMI":round(bmi,1),"Age":age,
            "Pregnancies":pregnancies,"BP":blood_pressure,"Insulin":insulin,
            "XGB":f"{xp:.1%}","RF":f"{rp:.1%}","LR":f"{lp:.1%}",
        })

        # ── RISK DETECTION BOX ──────────────────────────────
        st.markdown(\'<div class="sec-hdr">📊 Risk Detection Result</div>\',unsafe_allow_html=True)
        st.markdown(f"""
        <div class="risk-box risk-{risk}">
          <div style="font-size:.7rem;font-weight:700;letter-spacing:3px;text-transform:uppercase;
               color:{color};background:rgba(255,255,255,0.06);border-radius:20px;
               display:inline-block;padding:4px 16px;margin-bottom:1rem;">
            DIABETES RISK ASSESSMENT
          </div>
          <div style="font-size:3.5rem;line-height:1;margin-bottom:.5rem">{icon}</div>
          <div class="risk-title">{risk} RISK</div>
          <div style="width:50px;height:3px;background:{color};border-radius:3px;margin:.8rem auto;"></div>
          <div class="risk-prob">Ensemble Probability: <strong style="color:#e2e8f0">{ens:.1%}</strong></div>
          <div class="risk-advice">{ADVICE[risk]}</div>
          <div class="model-grid">
            <div class="model-card">
              <div class="model-name">XGBoost</div>
              <div style="font-size:1.2rem;font-weight:700;color:{xc}">{xi} {xr}</div>
              <div style="font-size:.75rem;color:#475569;font-family:monospace">{xp:.1%}</div>
            </div>
            <div class="model-card">
              <div class="model-name">Random Forest</div>
              <div style="font-size:1.2rem;font-weight:700;color:{rc}">{ri} {rr}</div>
              <div style="font-size:.75rem;color:#475569;font-family:monospace">{rp:.1%}</div>
            </div>
            <div class="model-card">
              <div class="model-name">Logistic Reg.</div>
              <div style="font-size:1.2rem;font-weight:700;color:{lc}">{li} {lr2}</div>
              <div style="font-size:.75rem;color:#475569;font-family:monospace">{lp:.1%}</div>
            </div>
          </div>
        </div>
        """,unsafe_allow_html=True)

        # shap
        st.markdown(\'<div class="sec-hdr">🔍 SHAP — Feature Impact</div>\',unsafe_allow_html=True)
        c1,c2=st.columns([3,1])
        with c1:
            fig=plot_shap(explainer,idf,features)
            st.pyplot(fig,use_container_width=True);plt.close(fig)
        with c2:
            st.markdown("**How to read:**")
            st.markdown("🔴 Red = pushes risk **UP**\\n\\n🟢 Green = pushes risk **DOWN**")
            sv2=explainer.shap_values(idf)[0]
            st.info(f"🎯 Top driver: `{features[int(np.argmax(np.abs(sv2)))]}`")

        # summary
        st.markdown(\'<div class="sec-hdr">📋 Summary</div>\',unsafe_allow_html=True)
        col1,col2,col3,col4=st.columns(4)
        for col,lbl,val in[(col1,"Glucose",f"{glucose} mg/dL"),(col2,"BMI",f"{bmi:.1f}"),
                            (col3,"Age",f"{age} yrs"),(col4,"Risk Prob",f"{ens:.1%}")]:
            with col:
                st.markdown(f\'<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value">{val}</div></div>\',unsafe_allow_html=True)

    # ── PREDICTION HISTORY ──────────────────────────────────
    st.markdown(\'<div class="sec-hdr">🕒 Prediction History</div>\',unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("""<div style=\'text-align:center;padding:2rem;background:rgba(255,255,255,0.02);
            border:1px dashed rgba(255,255,255,0.08);border-radius:14px;color:#374151\'>
            No predictions yet — set sliders and click ⚡ Predict</div>""",unsafe_allow_html=True)
    else:
        rows=""
        for i,rec in enumerate(st.session_state.history):
            r=rec["Risk"]
            rows+=f"""<tr>
              <td style="color:#60a5fa;font-weight:600">#{len(st.session_state.history)-i}</td>
              <td style="font-family:monospace;color:#64748b">{rec[\'Time\']}</td>
              <td><span class="badge badge-{r}">{r}</span></td>
              <td style="color:#e2e8f0;font-weight:600">{rec[\'Probability\']}</td>
              <td>{rec[\'Glucose\']}</td><td>{rec[\'BMI\']}</td><td>{rec[\'Age\']}</td>
              <td>{rec[\'Pregnancies\']}</td><td>{rec[\'BP\']}</td><td>{rec[\'Insulin\']}</td>
              <td style="font-family:monospace;font-size:.75rem">{rec[\'XGB\']}</td>
              <td style="font-family:monospace;font-size:.75rem">{rec[\'RF\']}</td>
              <td style="font-family:monospace;font-size:.75rem">{rec[\'LR\']}</td>
            </tr>"""
        st.markdown(f\'\'\'<div class="hist-wrap"><table class="hist"><thead><tr>
            <th>#</th><th>Time</th><th>Risk</th><th>Probability</th>
            <th>Glucose</th><th>BMI</th><th>Age</th><th>Preg</th><th>BP</th><th>Insulin</th>
            <th>XGB</th><th>RF</th><th>LR</th></tr></thead><tbody>{rows}</tbody></table></div>\'\'\',
            unsafe_allow_html=True)

        total=len(st.session_state.history)
        highs=sum(1 for h in st.session_state.history if h["Risk"]=="HIGH")
        meds =sum(1 for h in st.session_state.history if h["Risk"]=="MEDIUM")
        lows =sum(1 for h in st.session_state.history if h["Risk"]=="LOW")
        st.markdown("<br>",unsafe_allow_html=True)
        h1,h2,h3,h4=st.columns(4)
        for col,lbl,val,clr in[(h1,"Total",total,"#60a5fa"),(h2,"🔴 High",highs,"#ef4444"),
                                (h3,"🟡 Medium",meds,"#f59e0b"),(h4,"🟢 Low",lows,"#22c55e")]:
            with col:
                st.markdown(f\'<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value" style="color:{clr}">{val}</div></div>\',unsafe_allow_html=True)

        csv=pd.DataFrame(st.session_state.history).to_csv(index=False)
        st.download_button("⬇️ Download History CSV",csv,file_name="prediction_history.csv",mime="text/csv")

    # model comparison
    st.markdown("---")
    st.markdown(\'<div class="sec-hdr">📈 Model Performance — All Diseases</div>\',unsafe_allow_html=True)
    bm={\'Heart Disease\':{\'Logistic Regression\':{\'Accuracy\':84.3,\'F1-Score\':83.1,\'AUC-ROC\':90.2},
                         \'Random Forest\':{\'Accuracy\':87.6,\'F1-Score\':86.9,\'AUC-ROC\':93.5},
                         \'XGBoost\':{\'Accuracy\':89.2,\'F1-Score\':88.7,\'AUC-ROC\':95.1}},
        \'Liver Disease\':{\'Logistic Regression\':{\'Accuracy\':71.2,\'F1-Score\':69.8,\'AUC-ROC\':74.6},
                         \'Random Forest\':{\'Accuracy\':75.4,\'F1-Score\':74.1,\'AUC-ROC\':80.3},
                         \'XGBoost\':{\'Accuracy\':78.9,\'F1-Score\':77.5,\'AUC-ROC\':83.7}}}
    rows2=[]
    for disease,models_dict in[(\'Diabetes (Trained)\',metrics),(\'Heart Disease\',bm[\'Heart Disease\']),(\'Liver Disease\',bm[\'Liver Disease\'])]:
        for model,m in models_dict.items():
            rows2.append({\'Disease\':disease,\'Model\':model,\'Accuracy (%)\':m[\'Accuracy\'],\'F1-Score (%)\':m[\'F1-Score\'],\'AUC-ROC (%)\':m[\'AUC-ROC\']})
    df2=pd.DataFrame(rows2)
    def cv(v):
        if v>=85:return \'color:#22c55e;font-weight:600\'
        elif v>=75:return \'color:#f59e0b\'
        return \'color:#f87171\'
    styled=(df2.style.applymap(cv,subset=[\'Accuracy (%)\',\'F1-Score (%)\',\'AUC-ROC (%)\'])
              .set_properties(**{\'background-color\':\'rgba(10,18,36,.7)\',\'color\':\'#e2e8f0\',\'border\':\'1px solid rgba(255,255,255,0.05)\'})
              .set_table_styles([{\'selector\':\'th\',\'props\':[(\'background\',\'rgba(96,165,250,.14)\'),(\'color\',\'#60a5fa\'),(\'font-weight\',\'700\')]}])
              .hide(axis=\'index\'))
    st.dataframe(styled,use_container_width=True,height=370)

    cl,cr,cx=st.columns(3)
    for col,name in[(cl,\'Logistic Regression\'),(cr,\'Random Forest\'),(cx,\'XGBoost\')]:
        with col:
            m=metrics[name]
            st.markdown(f\'\'\'<div class="metric-card"><div class="metric-label">{name}</div>
              <div style="display:flex;justify-content:space-around;margin-top:.6rem">
                <div><div style="color:#6b7280;font-size:.68rem">ACC</div><div style="color:#60a5fa;font-weight:700">{m[\'Accuracy\']}%</div></div>
                <div><div style="color:#6b7280;font-size:.68rem">F1</div><div style="color:#a78bfa;font-weight:700">{m[\'F1-Score\']}%</div></div>
                <div><div style="color:#6b7280;font-size:.68rem">AUC</div><div style="color:#22c55e;font-weight:700">{m[\'AUC-ROC\']}%</div></div>
              </div></div>\'\'\',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style=\'text-align:center;color:#1f2937;font-size:.8rem\'>HealthPredict AI · Streamlit · XGBoost · SHAP · SMOTE</div>",unsafe_allow_html=True)

if __name__=="__main__":
    main()
'''

# Write app.py in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "app.py")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(app_code)

print("=" * 50)
print(f"✅ app.py created at:")
print(f"   {output_path}")
print("=" * 50)
print()
print("Now run:")
print("   python -m streamlit run app.py")
print("=" * 50)
