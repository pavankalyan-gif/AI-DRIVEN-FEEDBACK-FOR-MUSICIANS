"""
AI Music Feedback - Visualization Module (Cross-Platform)
Generates 10 training/testing charts. Works on Linux and Windows.
"""
import warnings, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
warnings.filterwarnings("ignore")

import sys
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path: sys.path.insert(0, str(_src))
from paths import PLOTS_DIR, ROOT

PALETTE  = ["#7C3AED","#06B6D4","#10B981","#F59E0B","#EF4444","#8B5CF6","#EC4899","#14B8A6"]
DARK_BG  = "#1E1B2E"; CARD_BG = "#2A2640"; TEXT_COL = "#E2E8F0"; GRID_COL = "#3D3757"

import matplotlib as _mpl
_MPL_NEW = tuple(int(x) for x in _mpl.__version__.split(".")[:2]) >= (3, 9)

def _boxplot(ax, data, labels, **kwargs):
    """Version-safe boxplot: matplotlib >=3.9 renamed labels -> tick_labels."""
    if _MPL_NEW:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    else:
        return ax.boxplot(data, labels=labels, **kwargs)

def _style():
    plt.rcParams.update({
        "figure.facecolor":DARK_BG,"axes.facecolor":CARD_BG,
        "axes.edgecolor":GRID_COL,"axes.labelcolor":TEXT_COL,
        "xtick.color":TEXT_COL,"ytick.color":TEXT_COL,
        "text.color":TEXT_COL,"grid.color":GRID_COL,
        "grid.alpha":0.4,"font.size":11,
        "axes.titlesize":13,"axes.titleweight":"bold","figure.dpi":100,
    })

def _save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(str(path), bbox_inches="tight", facecolor=DARK_BG, dpi=130)
    plt.close(fig)
    print(f"  ✓ {name}")
    return str(path)

def plot_dataset_overview(audio_df, logs_df, stats, **kw):
    _style()
    fig = plt.figure(figsize=(20,14))
    fig.suptitle("Dataset Overview — AI Music Feedback", fontsize=18, fontweight="bold", color=TEXT_COL, y=0.98)
    gs = gridspec.GridSpec(3,4, figure=fig, hspace=0.5, wspace=0.42)
    ax1=fig.add_subplot(gs[0,0])
    labels=["NSynth","MAESTRO","GuitarSet","Synthetic\nAudio","Synthetic\nLogs"]
    vals=[stats.get("nsynth_samples",0),stats.get("maestro_samples",0),stats.get("guitarset_samples",0),stats.get("synthetic_audio",0),stats.get("synthetic_logs",0)]
    bars=ax1.bar(labels,vals,color=PALETTE[:5],edgecolor="white",linewidth=0.5,zorder=3)
    ax1.set_title("Samples per Dataset"); ax1.set_ylabel("Count"); ax1.grid(axis="y",zorder=0)
    for b,v in zip(bars,vals): ax1.text(b.get_x()+b.get_width()/2,b.get_height()+5,str(v),ha="center",va="bottom",fontsize=9,color=TEXT_COL,fontweight="bold")
    ax2=fig.add_subplot(gs[0,1])
    sc=audio_df["skill_level"].value_counts()
    ax2.pie(sc.values,labels=sc.index,colors=PALETTE[:len(sc)],autopct="%1.1f%%",startangle=90,wedgeprops={"edgecolor":"white","linewidth":1.5})
    ax2.set_title("Skill Level Distribution")
    ax3=fig.add_subplot(gs[0,2])
    ic=audio_df["instrument"].value_counts().head(10)
    ax3.barh(ic.index[::-1],ic.values[::-1],color=PALETTE[2],edgecolor="white",linewidth=0.4,zorder=3)
    ax3.set_title("Top Instruments"); ax3.set_xlabel("Count"); ax3.grid(axis="x",zorder=0)
    ax4=fig.add_subplot(gs[0,3])
    src=audio_df["source"].value_counts()
    ax4.bar(src.index,src.values,color=PALETTE[3:3+len(src)],edgecolor="white",linewidth=0.5,zorder=3)
    ax4.set_title("Audio Features by Source"); ax4.set_ylabel("Count"); ax4.grid(axis="y",zorder=0)
    plt.setp(ax4.get_xticklabels(),rotation=30,ha="right",fontsize=8)
    ax5=fig.add_subplot(gs[1,0:2])
    for skill,col in zip(["Beginner","Intermediate","Advanced"],PALETTE[:3]):
        sub=audio_df[audio_df["skill_level"]==skill]["quality_score"].dropna()
        ax5.hist(sub,bins=30,alpha=0.65,color=col,label=skill,edgecolor="none")
    ax5.set_title("Quality Score Distribution by Skill Level"); ax5.set_xlabel("Quality Score (0-100)"); ax5.set_ylabel("Count"); ax5.legend(); ax5.grid(alpha=0.3)
    ax6=fig.add_subplot(gs[1,2:4])
    for src_,col in zip(audio_df["source"].unique(),PALETTE):
        sub=audio_df[audio_df["source"]==src_]["pitch_accuracy"].dropna()
        ax6.hist(sub,bins=30,alpha=0.6,color=col,label=src_,edgecolor="none")
    ax6.set_title("Pitch Accuracy by Source"); ax6.set_xlabel("Pitch Accuracy (0-1)"); ax6.set_ylabel("Count"); ax6.legend(); ax6.grid(alpha=0.3)
    ax7=fig.add_subplot(gs[2,0:2])
    if "alignment_score" in logs_df.columns:
        ax7.hist(logs_df["alignment_score"].dropna(),bins=30,color=PALETTE[4],edgecolor="none",alpha=0.85,zorder=3)
        ax7.axvline(0.75,color=PALETTE[2],ls="--",lw=1.5,label="High (0.75)")
        ax7.axvline(0.45,color=PALETTE[3],ls="--",lw=1.5,label="Low (0.45)")
    ax7.set_title("Practice Log Alignment Scores"); ax7.set_xlabel("Alignment Score"); ax7.set_ylabel("Count"); ax7.legend(); ax7.grid(alpha=0.3)
    ax8=fig.add_subplot(gs[2,2:4])
    if "duration_min" in logs_df.columns:
        ax8.hist(logs_df["duration_min"].dropna().clip(0,120),bins=30,color=PALETTE[5],edgecolor="none",alpha=0.85)
    ax8.set_title("Session Duration Distribution"); ax8.set_xlabel("Duration (min)"); ax8.set_ylabel("Count"); ax8.grid(alpha=0.3)
    return _save(fig,"01_dataset_overview.png")

def plot_correlation_heatmap(audio_df):
    _style()
    num_cols=[c for c in audio_df.select_dtypes(include=[np.number]).columns
              if "mfcc" not in c and "chroma" not in c and audio_df[c].notna().sum()>50][:18]
    corr=audio_df[num_cols].corr()
    fig,ax=plt.subplots(figsize=(16,13))
    fig.suptitle("Feature Correlation Heatmap",fontsize=16,fontweight="bold",color=TEXT_COL,y=1.01)
    mask=np.triu(np.ones_like(corr,dtype=bool))
    cmap=sns.diverging_palette(280,150,s=80,l=40,as_cmap=True)
    sns.heatmap(corr,mask=mask,cmap=cmap,center=0,vmin=-1,vmax=1,annot=True,fmt=".2f",
                linewidths=0.5,ax=ax,annot_kws={"size":7,"color":TEXT_COL},cbar_kws={"shrink":0.7})
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha="right",fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontsize=8)
    ax.set_facecolor(CARD_BG)
    return _save(fig,"02_correlation_heatmap.png")

def plot_quality_scorer(scorer_metrics):
    _style()
    fig=plt.figure(figsize=(20,12))
    fig.suptitle("Quality Scorer — Training & Testing Analysis (GradientBoosting)",fontsize=16,fontweight="bold",color=TEXT_COL,y=0.99)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)
    y_test=np.array(scorer_metrics["y_test"]); y_pred=np.array(scorer_metrics["y_pred"])
    y_train=np.array(scorer_metrics["y_train"]); y_tp=np.array(scorer_metrics["y_train_pred"])
    residuals=y_test-y_pred
    ax1=fig.add_subplot(gs[0,0])
    ax1.scatter(y_test,y_pred,alpha=0.55,color=PALETTE[0],s=20,edgecolors="none")
    mn,mx=min(y_test.min(),y_pred.min())-2,max(y_test.max(),y_pred.max())+2
    ax1.plot([mn,mx],[mn,mx],"--",color=PALETTE[2],lw=1.5,label="Perfect fit")
    ax1.set_xlabel("Actual"); ax1.set_ylabel("Predicted"); ax1.set_title(f"Actual vs Predicted (Test)\nMAE={scorer_metrics['test_mae']:.2f}  R²={scorer_metrics['test_r2']:.3f}"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2=fig.add_subplot(gs[0,1])
    ax2.scatter(y_train,y_tp,alpha=0.35,color=PALETTE[1],s=15,edgecolors="none")
    mn2,mx2=min(y_train.min(),y_tp.min())-2,max(y_train.max(),y_tp.max())+2
    ax2.plot([mn2,mx2],[mn2,mx2],"--",color=PALETTE[2],lw=1.5)
    ax2.set_xlabel("Actual"); ax2.set_ylabel("Predicted"); ax2.set_title(f"Actual vs Predicted (Train)\nMAE={scorer_metrics['train_mae']:.2f}  R²={scorer_metrics['train_r2']:.3f}"); ax2.grid(alpha=0.3)
    ax3=fig.add_subplot(gs[0,2])
    ax3.scatter(y_pred,residuals,alpha=0.55,color=PALETTE[3],s=20,edgecolors="none")
    ax3.axhline(0,color=PALETTE[2],ls="--",lw=1.5)
    ax3.set_xlabel("Predicted"); ax3.set_ylabel("Residual"); ax3.set_title("Residuals vs Predicted"); ax3.grid(alpha=0.3)
    ax4=fig.add_subplot(gs[1,0])
    ax4.hist(residuals,bins=35,color=PALETTE[4],edgecolor="none",alpha=0.85)
    ax4.axvline(0,color=PALETTE[2],ls="--",lw=1.5); ax4.axvline(residuals.mean(),color=PALETTE[1],ls=":",lw=1.5,label=f"Mean={residuals.mean():.2f}")
    ax4.set_xlabel("Residual"); ax4.set_ylabel("Frequency"); ax4.set_title("Error Distribution"); ax4.legend(); ax4.grid(alpha=0.3)
    ax5=fig.add_subplot(gs[1,1])
    mb={"Train MAE":scorer_metrics["train_mae"],"Test MAE":scorer_metrics["test_mae"],"CV MAE":scorer_metrics["cv_mae_mean"]}
    bars=ax5.bar(mb.keys(),mb.values(),color=PALETTE[:3],edgecolor="white",linewidth=0.6,zorder=3)
    ax5.errorbar(2,scorer_metrics["cv_mae_mean"],yerr=scorer_metrics["cv_mae_std"]*2,fmt="none",color="white",capsize=6,lw=2)
    ax5.set_title("MAE: Train / Test / CV (5-fold)"); ax5.set_ylabel("MAE (points)"); ax5.grid(axis="y",zorder=0)
    for b,v in zip(bars,mb.values()): ax5.text(b.get_x()+b.get_width()/2,b.get_height()+0.1,f"{v:.2f}",ha="center",va="bottom",fontsize=10,color=TEXT_COL,fontweight="bold")
    ax6=fig.add_subplot(gs[1,2])
    r2v={"Train R²":scorer_metrics["train_r2"],"Test R²":scorer_metrics["test_r2"]}
    bars6=ax6.bar(r2v.keys(),r2v.values(),color=[PALETTE[0],PALETTE[1]],edgecolor="white",linewidth=0.6,zorder=3)
    ax6.set_ylim(0,1.05); ax6.set_title("R² Score: Train vs Test"); ax6.set_ylabel("R²"); ax6.grid(axis="y",zorder=0)
    for b,v in zip(bars6,r2v.values()): ax6.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f"{v:.3f}",ha="center",va="bottom",fontsize=11,color=TEXT_COL,fontweight="bold")
    return _save(fig,"03_quality_scorer_analysis.png")

def plot_skill_classifier(clf_metrics):
    _style()
    fig=plt.figure(figsize=(20,12))
    fig.suptitle("Skill Level Classifier — Analysis (RandomForest)",fontsize=16,fontweight="bold",color=TEXT_COL,y=0.99)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.40)
    classes=clf_metrics["classes"]; cm=np.array(clf_metrics["confusion_matrix"]); report=clf_metrics["classification_report"]
    cmap=sns.light_palette(PALETTE[0],as_cmap=True)
    ax1=fig.add_subplot(gs[0,0])
    sns.heatmap(cm,annot=True,fmt="d",cmap=cmap,ax=ax1,xticklabels=classes,yticklabels=classes,linewidths=0.5,linecolor=DARK_BG,annot_kws={"size":14,"fontweight":"bold","color":TEXT_COL})
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual"); ax1.set_title(f"Confusion Matrix\nAccuracy={clf_metrics['accuracy']:.3f}")
    ax2=fig.add_subplot(gs[0,1])
    cm_norm=cm.astype(float)/cm.sum(axis=1,keepdims=True)
    sns.heatmap(cm_norm,annot=True,fmt=".2f",cmap=cmap,ax=ax2,xticklabels=classes,yticklabels=classes,linewidths=0.5,linecolor=DARK_BG,annot_kws={"size":13,"fontweight":"bold","color":TEXT_COL},vmin=0,vmax=1)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual"); ax2.set_title("Confusion Matrix (Normalised)")
    ax3=fig.add_subplot(gs[0,2])
    x=np.arange(len(classes)); w=0.25
    prec=[report[c]["precision"] for c in classes]; rec=[report[c]["recall"] for c in classes]; f1=[report[c]["f1-score"] for c in classes]
    ax3.bar(x-w,prec,w,label="Precision",color=PALETTE[0],edgecolor="white",lw=0.5,zorder=3)
    ax3.bar(x,rec,w,label="Recall",color=PALETTE[1],edgecolor="white",lw=0.5,zorder=3)
    ax3.bar(x+w,f1,w,label="F1-Score",color=PALETTE[2],edgecolor="white",lw=0.5,zorder=3)
    ax3.set_xticks(x); ax3.set_xticklabels(classes); ax3.set_ylim(0,1.1); ax3.set_title("Per-Class Metrics"); ax3.legend(); ax3.grid(axis="y",zorder=0)
    ax4=fig.add_subplot(gs[1,0])
    cv_m=clf_metrics["cv_acc_mean"]; cv_s=clf_metrics["cv_acc_std"]
    ax4.bar(["CV Accuracy","Test Accuracy"],[cv_m,clf_metrics["accuracy"]],color=[PALETTE[3],PALETTE[4]],edgecolor="white",lw=0.6,zorder=3)
    ax4.errorbar(0,cv_m,yerr=cv_s*2,fmt="none",color="white",capsize=8,lw=2)
    ax4.set_ylim(0,1.15); ax4.set_title("CV vs Test Accuracy"); ax4.set_ylabel("Accuracy"); ax4.grid(axis="y",zorder=0)
    ax4.text(0,cv_m+0.01,f"{cv_m:.3f}±{cv_s:.3f}",ha="center",fontsize=9,color=TEXT_COL,fontweight="bold")
    ax4.text(1,clf_metrics["accuracy"]+0.01,f"{clf_metrics['accuracy']:.3f}",ha="center",fontsize=9,color=TEXT_COL,fontweight="bold")
    ax5=fig.add_subplot(gs[1,1])
    supports=[int(report[c]["support"]) for c in classes]
    ax5.bar(classes,supports,color=PALETTE[:len(classes)],edgecolor="white",lw=0.5,zorder=3)
    ax5.set_title("Test Set Class Support"); ax5.set_ylabel("Samples"); ax5.grid(axis="y",zorder=0)
    for i,(cls,s) in enumerate(zip(classes,supports)): ax5.text(i,s+0.5,str(s),ha="center",va="bottom",fontsize=10,color=TEXT_COL,fontweight="bold")
    ax6=fig.add_subplot(gs[1,2])
    cats=["macro avg","weighted avg"]; x2=np.arange(len(cats)); w2=0.25
    p_v=[report[c]["precision"] for c in cats]; r_v=[report[c]["recall"] for c in cats]; f_v=[report[c]["f1-score"] for c in cats]
    ax6.bar(x2-w2,p_v,w2,label="Precision",color=PALETTE[0],edgecolor="white",lw=0.5,zorder=3)
    ax6.bar(x2,r_v,w2,label="Recall",color=PALETTE[1],edgecolor="white",lw=0.5,zorder=3)
    ax6.bar(x2+w2,f_v,w2,label="F1",color=PALETTE[2],edgecolor="white",lw=0.5,zorder=3)
    ax6.set_xticks(x2); ax6.set_xticklabels(["Macro Avg","Weighted Avg"]); ax6.set_ylim(0,1.1); ax6.set_title("Macro & Weighted Averages"); ax6.legend(); ax6.grid(axis="y",zorder=0)
    return _save(fig,"04_skill_classifier_analysis.png")

def plot_feature_importance(scorer, clf):
    _style()
    fig=plt.figure(figsize=(20,10))
    fig.suptitle("Feature Importance Analysis",fontsize=16,fontweight="bold",color=TEXT_COL,y=0.99)
    gs=gridspec.GridSpec(1,2,figure=fig,wspace=0.4)
    import matplotlib.colors as mc
    def _fi_bar(ax,fi_df,title,color_base):
        top=fi_df.head(20)
        colors=[mc.to_rgba(color_base,alpha=0.5+0.5*(1-i/20)) for i in range(len(top))]
        ax.barh(top["feature"][::-1],top["importance"][::-1],color=colors[::-1],edgecolor="white",linewidth=0.4,zorder=3)
        ax.set_title(title); ax.set_xlabel("Importance Score"); ax.grid(axis="x",zorder=0)
        for b in ax.patches:
            w=b.get_width()
            ax.text(w+0.0002,b.get_y()+b.get_height()/2,f"{w:.4f}",va="center",fontsize=7,color=TEXT_COL)
    ax1=fig.add_subplot(gs[0,0]); _fi_bar(ax1,scorer.feature_importance(),"Quality Scorer — Top-20 Features (GradientBoosting)",PALETTE[0])
    ax2=fig.add_subplot(gs[0,1])
    rf_fi=pd.DataFrame({"feature":clf.feature_cols,"importance":clf.pipeline.named_steps["model"].feature_importances_}).sort_values("importance",ascending=False)
    _fi_bar(ax2,rf_fi,"Skill Classifier — Top-20 Features (RandomForest)",PALETTE[2])
    return _save(fig,"05_feature_importance.png")

def plot_model_comparison(scorer_metrics, clf_metrics):
    _style()
    fig=plt.figure(figsize=(20,8))
    fig.suptitle("Model Performance Comparison Dashboard",fontsize=16,fontweight="bold",color=TEXT_COL,y=1.01)
    gs=gridspec.GridSpec(1,3,figure=fig,wspace=0.42)
    ax1=fig.add_subplot(gs[0,0])
    regs={"Train MAE":scorer_metrics["train_mae"],"Test MAE":scorer_metrics["test_mae"],"Test RMSE":scorer_metrics["test_rmse"],"CV MAE":scorer_metrics["cv_mae_mean"]}
    bars1=ax1.bar(regs.keys(),regs.values(),color=PALETTE[:4],edgecolor="white",lw=0.6,zorder=3)
    ax1.set_title("Regression Model (Quality Scorer)\nGradientBoosting"); ax1.set_ylabel("Error (pts)"); ax1.grid(axis="y",zorder=0)
    plt.setp(ax1.get_xticklabels(),rotation=25,ha="right",fontsize=9)
    for b,v in zip(bars1,regs.values()): ax1.text(b.get_x()+b.get_width()/2,b.get_height()+0.05,f"{v:.2f}",ha="center",va="bottom",fontsize=9,color=TEXT_COL,fontweight="bold")
    ax2=fig.add_subplot(gs[0,1])
    clfs={"Test Accuracy":clf_metrics["accuracy"],"CV Accuracy":clf_metrics["cv_acc_mean"],"Macro F1":clf_metrics["classification_report"]["macro avg"]["f1-score"],"Weighted F1":clf_metrics["classification_report"]["weighted avg"]["f1-score"]}
    bars2=ax2.bar(clfs.keys(),clfs.values(),color=PALETTE[4:8],edgecolor="white",lw=0.6,zorder=3)
    ax2.set_ylim(0,1.15); ax2.set_title("Classification Model (Skill Level)\nRandomForest"); ax2.set_ylabel("Score"); ax2.grid(axis="y",zorder=0)
    plt.setp(ax2.get_xticklabels(),rotation=25,ha="right",fontsize=9)
    for b,v in zip(bars2,clfs.values()): ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f"{v:.3f}",ha="center",va="bottom",fontsize=9,color=TEXT_COL,fontweight="bold")
    ax3=fig.add_subplot(gs[0,2]); ax3.axis("off")
    lines=["━"*27,"  MODEL SUMMARY","━"*27,"",
           "  GradientBoosting Regressor",
           f"  Train samples  : {scorer_metrics['train_samples']}",
           f"  Test samples   : {scorer_metrics['test_samples']}",
           f"  Features used  : {scorer_metrics['n_features']}",
           f"  Train MAE      : {scorer_metrics['train_mae']:.2f} pts",
           f"  Test  MAE      : {scorer_metrics['test_mae']:.2f} pts",
           f"  Test  RMSE     : {scorer_metrics['test_rmse']:.2f} pts",
           f"  Train R²       : {scorer_metrics['train_r2']:.3f}",
           f"  Test  R²       : {scorer_metrics['test_r2']:.3f}",
           f"  CV MAE (5-fold): {scorer_metrics['cv_mae_mean']:.2f}±{scorer_metrics['cv_mae_std']:.2f}",
           "","  RandomForest Classifier",
           f"  Train samples  : {clf_metrics['train_samples']}",
           f"  Test samples   : {clf_metrics['test_samples']}",
           f"  Test Accuracy  : {clf_metrics['accuracy']:.3f}",
           f"  CV  Accuracy   : {clf_metrics['cv_acc_mean']:.3f}±{clf_metrics['cv_acc_std']:.3f}",
           f"  Macro F1       : {clf_metrics['classification_report']['macro avg']['f1-score']:.3f}",
           "","━"*27]
    ax3.text(0.05,0.97,"\n".join(lines),transform=ax3.transAxes,fontfamily="monospace",fontsize=9.5,color=TEXT_COL,va="top",ha="left",bbox=dict(boxstyle="round,pad=0.5",facecolor=CARD_BG,edgecolor=PALETTE[0],linewidth=1.5))
    return _save(fig,"06_model_comparison.png")

def plot_acoustic_features(audio_df):
    _style()
    fig=plt.figure(figsize=(20,14))
    fig.suptitle("Acoustic Feature Distributions",fontsize=16,fontweight="bold",color=TEXT_COL,y=0.99)
    gs=gridspec.GridSpec(3,2,figure=fig,hspace=0.5,wspace=0.38)
    ax1=fig.add_subplot(gs[0,:])
    mfcc_cols=[f"mfcc_{j}_mean" for j in range(13) if f"mfcc_{j}_mean" in audio_df.columns]
    if mfcc_cols:
        mb=audio_df.groupby("skill_level")[mfcc_cols].mean()
        cmap2=sns.diverging_palette(280,150,s=80,l=40,as_cmap=True)
        sns.heatmap(mb,cmap=cmap2,center=0,annot=True,fmt=".1f",ax=ax1,linewidths=0.3,linecolor=DARK_BG,annot_kws={"size":8,"color":TEXT_COL},cbar_kws={"shrink":0.6})
        ax1.set_title("Mean MFCC Coefficients by Skill Level"); ax1.set_xlabel("MFCC Coefficient"); ax1.set_ylabel("Skill Level")
    else:
        ax1.text(0.5,0.5,"MFCC data not available",ha="center",va="center",transform=ax1.transAxes,color=TEXT_COL); ax1.set_title("MFCC by Skill Level")
    ax2=fig.add_subplot(gs[1,0])
    chroma_cols=[f"chroma_{j}" for j in range(12) if f"chroma_{j}" in audio_df.columns]
    note_names=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    if chroma_cols:
        cm=audio_df[chroma_cols].mean()
        ax2.bar(note_names[:len(chroma_cols)],cm.values,color=PALETTE,edgecolor="white",lw=0.4,zorder=3)
    ax2.set_title("Mean Chroma Energy"); ax2.set_ylabel("Mean"); ax2.set_xlabel("Pitch Class"); ax2.grid(axis="y",zorder=0)
    ax3=fig.add_subplot(gs[1,1])
    skills=["Beginner","Intermediate","Advanced"]
    data3=[audio_df[audio_df["skill_level"]==s]["spectral_centroid"].dropna().values for s in skills]
    bp=_boxplot(ax3,data3,skills,patch_artist=True,medianprops=dict(color="white",lw=2))
    for patch,col in zip(bp["boxes"],PALETTE[:3]): patch.set_facecolor(col); patch.set_alpha(0.7)
    ax3.set_title("Spectral Centroid by Skill Level"); ax3.set_ylabel("Hz"); ax3.grid(axis="y",alpha=0.3)
    ax4=fig.add_subplot(gs[2,0])
    for i,(s,col) in enumerate(zip(skills,PALETTE[:3])):
        vals=audio_df[audio_df["skill_level"]==s]["loudness_consistency"].dropna()
        parts=ax4.violinplot([vals],positions=[i],showmedians=True,showextrema=True)
        for pc in parts["bodies"]: pc.set_facecolor(col); pc.set_alpha(0.7)
        parts["cmedians"].set_color("white"); parts["cmedians"].set_lw(2)
        parts["cbars"].set_color(col); parts["cmaxes"].set_color(col); parts["cmins"].set_color(col)
    ax4.set_xticks(range(len(skills))); ax4.set_xticklabels(skills)
    ax4.set_title("Loudness Consistency by Skill Level"); ax4.set_ylabel("Score (0-1)"); ax4.grid(axis="y",alpha=0.3)
    ax5=fig.add_subplot(gs[2,1])
    for s,col in zip(skills,PALETTE[:3]):
        sub=audio_df[audio_df["skill_level"]==s]
        ax5.scatter(sub["zero_crossing_rate"],sub["quality_score"],alpha=0.4,color=col,s=18,label=s,edgecolors="none")
    ax5.set_xlabel("Zero Crossing Rate"); ax5.set_ylabel("Quality Score"); ax5.set_title("ZCR vs Quality Score"); ax5.legend(); ax5.grid(alpha=0.3)
    return _save(fig,"07_acoustic_features.png")

def plot_alignment_analysis(logs_df):
    _style()
    fig=plt.figure(figsize=(20,10))
    fig.suptitle("Feedback Alignment Analysis — Student vs Teacher",fontsize=16,fontweight="bold",color=TEXT_COL,y=0.99)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.4)
    ax1=fig.add_subplot(gs[0,0:2])
    if "alignment_score" in logs_df.columns:
        scores=logs_df["alignment_score"].dropna()
        ax1.hist(scores,bins=35,color=PALETTE[0],edgecolor="none",alpha=0.85,zorder=3)
        high=(scores>0.75).sum(); mod=((scores>=0.45)&(scores<=0.75)).sum(); low=(scores<0.45).sum()
        for thr,col,lbl in [(0.75,PALETTE[2],"High (>0.75)"),(0.45,PALETTE[3],"Low (<0.45)")]:
            ax1.axvline(thr,color=col,ls="--",lw=2,label=lbl)
        ax1.text(0.78,0.92,f"High: {high} ({high/len(scores)*100:.0f}%)",transform=ax1.transAxes,color=PALETTE[2],fontsize=10,fontweight="bold")
        ax1.text(0.78,0.84,f"Mod:  {mod}  ({mod/len(scores)*100:.0f}%)",transform=ax1.transAxes,color=TEXT_COL,fontsize=10)
        ax1.text(0.78,0.76,f"Low:  {low}  ({low/len(scores)*100:.0f}%)",transform=ax1.transAxes,color=PALETTE[3],fontsize=10)
    ax1.set_title("Alignment Score Distribution"); ax1.set_xlabel("Alignment Score"); ax1.set_ylabel("Count"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2=fig.add_subplot(gs[0,2])
    if "alignment_score" in logs_df.columns:
        scores=logs_df["alignment_score"].dropna()
        cats=["High (>0.75)","Moderate","Low (<0.45)"]
        vals=[(scores>0.75).sum(),((scores>=0.45)&(scores<=0.75)).sum(),(scores<0.45).sum()]
        ax2.pie(vals,labels=cats,colors=[PALETTE[2],PALETTE[1],PALETTE[4]],autopct="%1.1f%%",startangle=90,wedgeprops={"edgecolor":"white","linewidth":1.5})
    ax2.set_title("Alignment Category Breakdown")
    ax3=fig.add_subplot(gs[1,0])
    if "student_rating" in logs_df.columns and "teacher_rating" in logs_df.columns:
        ax3.scatter(logs_df["student_rating"].dropna(),logs_df["teacher_rating"].dropna(),alpha=0.45,color=PALETTE[5],s=20,edgecolors="none")
        ax3.plot([1,10],[1,10],"--",color=PALETTE[2],lw=1.5,label="Perfect agreement")
        ax3.set_xlabel("Student Rating"); ax3.set_ylabel("Teacher Rating"); ax3.set_title("Student vs Teacher Ratings"); ax3.legend(); ax3.grid(alpha=0.3)
    ax4=fig.add_subplot(gs[1,1])
    if "student_rating" in logs_df.columns and "teacher_rating" in logs_df.columns:
        gap=(logs_df["student_rating"]-logs_df["teacher_rating"]).dropna()
        ax4.hist(gap,bins=20,color=PALETTE[6],edgecolor="none",alpha=0.85,zorder=3)
        ax4.axvline(0,color=PALETTE[2],ls="--",lw=1.5)
        ax4.set_xlabel("Student - Teacher Rating"); ax4.set_ylabel("Count"); ax4.set_title(f"Rating Gap Distribution\nMean={gap.mean():.2f}, Std={gap.std():.2f}"); ax4.grid(alpha=0.3)
    ax5=fig.add_subplot(gs[1,2])
    if "alignment_score" in logs_df.columns and "skill_level" in logs_df.columns:
        sa=logs_df.groupby("skill_level")["alignment_score"].mean().sort_values()
        cols=[PALETTE[i] for i in range(len(sa))]
        bars=ax5.bar(sa.index,sa.values,color=cols,edgecolor="white",lw=0.5,zorder=3)
        ax5.set_title("Mean Alignment by Skill Level"); ax5.set_ylabel("Mean Alignment"); ax5.set_ylim(0,1.05); ax5.grid(axis="y",zorder=0)
        for b,v in zip(bars,sa.values): ax5.text(b.get_x()+b.get_width()/2,v+0.01,f"{v:.3f}",ha="center",va="bottom",fontsize=9,color=TEXT_COL,fontweight="bold")
    return _save(fig,"08_alignment_analysis.png")

def plot_dataset_source_comparison(audio_df):
    _style()
    fig=plt.figure(figsize=(20,10))
    fig.suptitle("Feature Comparison Across Dataset Sources",fontsize=16,fontweight="bold",color=TEXT_COL,y=0.99)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.4)
    key_features=["quality_score","pitch_accuracy","loudness_consistency","spectral_centroid","zero_crossing_rate","tempo"]
    titles=["Quality Score","Pitch Accuracy","Loudness Consistency","Spectral Centroid (Hz)","Zero Crossing Rate","Tempo (BPM)"]
    sources=audio_df["source"].unique()
    for idx,(feat,title) in enumerate(zip(key_features,titles)):
        ax=fig.add_subplot(gs[idx//3,idx%3])
        if feat not in audio_df.columns: continue
        data=[audio_df[audio_df["source"]==s][feat].dropna().values for s in sources]
        lbs=[s.replace("_synthetic","*").replace("nsynth","NSynth").replace("guitarset","Guitar") for s in sources]
        bp=_boxplot(ax,data,lbs,patch_artist=True,medianprops=dict(color="white",lw=2))
        for patch,col in zip(bp["boxes"],PALETTE): patch.set_facecolor(col); patch.set_alpha(0.72)
        ax.set_title(title); ax.grid(axis="y",alpha=0.3); plt.setp(ax.get_xticklabels(),rotation=22,ha="right",fontsize=8)
    return _save(fig,"09_dataset_source_comparison.png")

def plot_summary_dashboard(audio_df, logs_df, scorer_metrics, clf_metrics, stats):
    _style()
    fig=plt.figure(figsize=(24,16)); fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("AI Music Feedback — Complete Summary Dashboard",fontsize=20,fontweight="bold",color=TEXT_COL,y=0.98)
    gs=gridspec.GridSpec(3,4,figure=fig,hspace=0.55,wspace=0.45)
    kpis=[("Total Training Samples",f"{stats['total_audio']:,}",PALETTE[0]),("Test MAE (Quality)",f"{scorer_metrics['test_mae']:.2f} pts",PALETTE[1]),("Skill Accuracy",f"{clf_metrics['accuracy']*100:.1f}%",PALETTE[2]),("Datasets Used","3 Real + Synthetic",PALETTE[3])]
    for i,(title,val,col) in enumerate(kpis):
        ax=fig.add_subplot(gs[0,i]); ax.set_facecolor(col+"33")
        for spine in ax.spines.values(): spine.set_edgecolor(col); spine.set_linewidth(2)
        ax.axis("off"); ax.text(0.5,0.62,val,transform=ax.transAxes,ha="center",va="center",fontsize=22,fontweight="bold",color=col)
        ax.text(0.5,0.25,title,transform=ax.transAxes,ha="center",va="center",fontsize=11,color=TEXT_COL)
    ax5=fig.add_subplot(gs[1,0:2])
    y_test=np.array(scorer_metrics["y_test"]); y_pred=np.array(scorer_metrics["y_pred"])
    ax5.scatter(y_test,y_pred,alpha=0.45,color=PALETTE[0],s=16,edgecolors="none")
    mn,mx=min(y_test.min(),y_pred.min())-2,max(y_test.max(),y_pred.max())+2
    ax5.plot([mn,mx],[mn,mx],"--",color=PALETTE[2],lw=1.5)
    ax5.set_xlabel("Actual Quality Score"); ax5.set_ylabel("Predicted Quality Score")
    ax5.set_title(f"Quality Scorer: Actual vs Predicted\nR²={scorer_metrics['test_r2']:.3f}  MAE={scorer_metrics['test_mae']:.2f}"); ax5.grid(alpha=0.3)
    ax6=fig.add_subplot(gs[1,2])
    cm=np.array(clf_metrics["confusion_matrix"]); classes=clf_metrics["classes"]
    cmap3=sns.light_palette(PALETTE[2],as_cmap=True)
    sns.heatmap(cm,annot=True,fmt="d",cmap=cmap3,ax=ax6,xticklabels=classes,yticklabels=classes,linewidths=0.4,linecolor=DARK_BG,annot_kws={"size":13,"fontweight":"bold","color":TEXT_COL})
    ax6.set_xlabel("Predicted"); ax6.set_ylabel("Actual"); ax6.set_title(f"Confusion Matrix\nAcc={clf_metrics['accuracy']:.3f}")
    ax7=fig.add_subplot(gs[1,3])
    fi=pd.read_csv(ROOT/"data"/"models"/"feature_importance.csv").head(10)
    ax7.barh(fi["feature"][::-1],fi["importance"][::-1],color=PALETTE[0],edgecolor="white",lw=0.4,zorder=3)
    ax7.set_title("Top-10 Feature Importances"); ax7.set_xlabel("Importance"); ax7.grid(axis="x",zorder=0)
    ax8=fig.add_subplot(gs[2,0:2])
    for s,col in zip(["Beginner","Intermediate","Advanced"],PALETTE[:3]):
        sub=audio_df[audio_df["skill_level"]==s]["quality_score"].dropna()
        ax8.hist(sub,bins=25,alpha=0.6,color=col,label=s,edgecolor="none")
    ax8.set_title("Quality Score Distribution by Skill Level"); ax8.set_xlabel("Quality Score"); ax8.set_ylabel("Count"); ax8.legend(); ax8.grid(alpha=0.3)
    ax9=fig.add_subplot(gs[2,2])
    if "alignment_score" in logs_df.columns:
        sc=logs_df["alignment_score"].dropna()
        ax9.hist(sc,bins=25,color=PALETTE[4],edgecolor="none",alpha=0.85,zorder=3)
        ax9.axvline(0.75,color=PALETTE[2],ls="--",lw=1.5); ax9.axvline(0.45,color=PALETTE[3],ls="--",lw=1.5)
    ax9.set_title("Feedback Alignment Scores"); ax9.set_xlabel("Alignment Score"); ax9.set_ylabel("Count"); ax9.grid(alpha=0.3)
    ax10=fig.add_subplot(gs[2,3])
    ds_labels=["NSynth","MAESTRO","GuitarSet","Synthetic\nAudio","Synthetic\nLogs"]
    ds_vals=[stats["nsynth_samples"],stats["maestro_samples"],stats["guitarset_samples"],stats["synthetic_audio"],stats["synthetic_logs"]]
    ax10.bar(ds_labels,ds_vals,color=PALETTE[:5],edgecolor="white",lw=0.5,zorder=3)
    ax10.set_title("Dataset Sample Counts"); ax10.set_ylabel("Samples"); ax10.grid(axis="y",zorder=0); plt.setp(ax10.get_xticklabels(),fontsize=8)
    return _save(fig,"10_summary_dashboard.png")

def generate_all_plots(audio_df, logs_df, stats, scorer, clf):
    print("\n"+"="*62); print("  GENERATING VISUALIZATIONS"); print("="*62)
    paths=[]
    paths.append(plot_dataset_overview(audio_df,logs_df,stats))
    paths.append(plot_correlation_heatmap(audio_df))
    paths.append(plot_quality_scorer(scorer.metrics))
    paths.append(plot_skill_classifier(clf.metrics))
    paths.append(plot_feature_importance(scorer,clf))
    paths.append(plot_model_comparison(scorer.metrics,clf.metrics))
    paths.append(plot_acoustic_features(audio_df))
    paths.append(plot_alignment_analysis(logs_df))
    paths.append(plot_dataset_source_comparison(audio_df))
    paths.append(plot_summary_dashboard(audio_df,logs_df,scorer.metrics,clf.metrics,stats))
    print(f"\n  ✓ {len(paths)} plots saved → {PLOTS_DIR}")
    return paths
