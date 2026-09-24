"""
AI Music Feedback — ML Model Module (Cross-Platform)
Works on Linux and Windows.
"""
import json, pickle, warnings
import numpy as np, pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, r2_score, mean_squared_error,
    accuracy_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")

import sys
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path: sys.path.insert(0, str(_src))
from paths import MODELS_DIR

EXCLUDE = {"source","instrument","skill_level","quality_score","detected_note",
           "pitch_in_tune","piece","focus_area","composer","title","genre","key",
           "split","student_self_report","teacher_feedback"}

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]

class QualityScorer:
    def __init__(self):
        self.pipeline=None; self.feature_cols=None; self.metrics={}
        self.model_path = MODELS_DIR/"quality_scorer.pkl"
    def train(self, df):
        self.feature_cols=get_feature_cols(df)
        X=df[self.feature_cols].values; y=df["quality_score"].values
        X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
        self.pipeline=Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
            ("model",GradientBoostingRegressor(n_estimators=200,max_depth=4,
             learning_rate=0.07,subsample=0.8,min_samples_leaf=3,random_state=42))])
        self.pipeline.fit(X_tr,y_tr)
        tr_pred=self.pipeline.predict(X_tr); te_pred=self.pipeline.predict(X_te)
        cv=cross_val_score(self.pipeline,X,y,cv=5,scoring="neg_mean_absolute_error")
        self.metrics={"train_mae":float(mean_absolute_error(y_tr,tr_pred)),
            "test_mae":float(mean_absolute_error(y_te,te_pred)),
            "train_r2":float(r2_score(y_tr,tr_pred)),
            "test_r2":float(r2_score(y_te,te_pred)),
            "test_rmse":float(np.sqrt(mean_squared_error(y_te,te_pred))),
            "cv_mae_mean":float(-cv.mean()),"cv_mae_std":float(cv.std()),
            "train_samples":int(len(X_tr)),"test_samples":int(len(X_te)),
            "n_features":int(len(self.feature_cols)),
            "y_test":y_te.tolist(),"y_pred":te_pred.tolist(),
            "y_train":y_tr.tolist(),"y_train_pred":tr_pred.tolist()}
        print(f"  QualityScorer  → test MAE={self.metrics['test_mae']:.2f}  R²={self.metrics['test_r2']:.3f}  CV-MAE={self.metrics['cv_mae_mean']:.2f}±{self.metrics['cv_mae_std']:.2f}")
        return self.metrics
    def predict(self,features):
        X=np.array([[features.get(c,0.0) for c in self.feature_cols]])
        return float(np.clip(self.pipeline.predict(X)[0],0,100))
    def feature_importance(self):
        imp=self.pipeline.named_steps["model"].feature_importances_
        return pd.DataFrame({"feature":self.feature_cols,"importance":imp}).sort_values("importance",ascending=False)
    def save(self):
        with open(self.model_path,"wb") as f:
            pickle.dump({"pipeline":self.pipeline,"feature_cols":self.feature_cols,"metrics":self.metrics},f)
    def load(self):
        with open(self.model_path,"rb") as f: d=pickle.load(f)
        self.pipeline=d["pipeline"]; self.feature_cols=d["feature_cols"]; self.metrics=d["metrics"]

class SkillLevelClassifier:
    LABELS=["Beginner","Intermediate","Advanced"]
    def __init__(self):
        self.pipeline=None; self.le=LabelEncoder()
        self.feature_cols=None; self.metrics={}
        self.model_path=MODELS_DIR/"skill_classifier.pkl"
    def train(self,df):
        df=df.dropna(subset=["skill_level"])
        self.feature_cols=get_feature_cols(df)
        X=df[self.feature_cols].values; y=self.le.fit_transform(df["skill_level"].values)
        X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        self.pipeline=Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
            ("model",RandomForestClassifier(n_estimators=300,max_depth=10,
             class_weight="balanced",random_state=42,n_jobs=-1))])
        self.pipeline.fit(X_tr,y_tr)
        y_pred=self.pipeline.predict(X_te)
        cv_acc=cross_val_score(self.pipeline,X,y,cv=5,scoring="accuracy")
        report=classification_report(y_te,y_pred,target_names=self.le.classes_,output_dict=True)
        self.metrics={"accuracy":float(accuracy_score(y_te,y_pred)),
            "cv_acc_mean":float(cv_acc.mean()),"cv_acc_std":float(cv_acc.std()),
            "confusion_matrix":confusion_matrix(y_te,y_pred).tolist(),
            "classification_report":report,"classes":list(self.le.classes_),
            "train_samples":int(len(X_tr)),"test_samples":int(len(X_te)),
            "y_test":y_te.tolist(),"y_pred":y_pred.tolist()}
        print(f"  SkillClassifier → accuracy={self.metrics['accuracy']:.3f}  CV={self.metrics['cv_acc_mean']:.3f}±{self.metrics['cv_acc_std']:.3f}")
        return self.metrics
    def predict(self,features):
        X=np.array([[features.get(c,0.0) for c in self.feature_cols]])
        idx=self.pipeline.predict(X)[0]; proba=self.pipeline.predict_proba(X)[0]
        return self.le.inverse_transform([idx])[0],{cls:float(p) for cls,p in zip(self.le.classes_,proba)}
    def save(self):
        with open(self.model_path,"wb") as f:
            pickle.dump({"pipeline":self.pipeline,"le":self.le,"feature_cols":self.feature_cols,"metrics":self.metrics},f)
    def load(self):
        with open(self.model_path,"rb") as f: d=pickle.load(f)
        self.pipeline=d["pipeline"]; self.le=d["le"]
        self.feature_cols=d["feature_cols"]; self.metrics=d["metrics"]

def train_all_models(df):
    print("\n[ML] Training Quality Scorer (GradientBoosting)...")
    scorer=QualityScorer(); scorer.train(df); scorer.save()
    print("\n[ML] Training Skill Classifier (RandomForest)...")
    clf=SkillLevelClassifier(); clf.train(df); clf.save()
    fi=scorer.feature_importance()
    fi.to_csv(MODELS_DIR/"feature_importance.csv",index=False)
    summary={"quality_scorer":scorer.metrics,"skill_classifier":clf.metrics}
    (MODELS_DIR/"model_summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(f"\n  ✓  Top-5 features: {', '.join(fi.head(5)['feature'].tolist())}")
    return {"scorer":scorer,"classifier":clf,"metrics":summary}
