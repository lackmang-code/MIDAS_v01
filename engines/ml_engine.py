# -*- coding: utf-8 -*-
"""
ml_engine.py
────────────
QSPR 머신러닝 모델 훈련 / 교차검증 / 예측 엔진

주요 함수
  split_data(X, y)                  : stratified train/val/test 분할
  train_all_models(X_tr, y_tr)      : GBR · RF · Ridge 3종 학습
  cross_validate(X_tr, y_tr)        : 5-fold CV (GBR 기준)
  evaluate(model, X, y, log_y)      : RMSE · MAE · R² 반환
  predict_ensemble(models, X_new)   : 앙상블 예측 + 불확실도
  calc_leverage(X_tr, X_new)        : Leverage 점수 (AD 판정)
  run_pipeline(X, y)                : 전체 파이프라인 1-call 실행

디자인 원칙
  - log(k) 변환 후 학습 → k 역변환 후 보고
  - stratified split: k 구간별 균등 분할
  - GBR 이 주력, RF·Ridge 는 비교 모델 + 불확실도 추정용
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 기본 하이퍼파라미터
# ─────────────────────────────────────────────────────────────────────────────
GBR_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=8,
    min_samples_leaf=5,
    random_state=42,
)
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
RIDGE_PARAMS = dict(alpha=10.0)


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────
def _k_bins(y: np.ndarray) -> np.ndarray:
    """k 값을 5개 구간으로 변환 (stratified split 용)"""
    bins = np.digitize(y, bins=[2.0, 3.0, 5.0, 10.0])
    return bins


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 분할
# ─────────────────────────────────────────────────────────────────────────────
def split_data(X: pd.DataFrame,
               y: pd.Series,
               test_size: float = 0.15,
               val_size:  float = 0.15,
               random_state: int = 42
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                          pd.Series,  pd.Series,  pd.Series]:
    """
    k 구간별 stratified split  →  train / val / test 반환

    Returns
    -------
    X_tr, X_val, X_te, y_tr, y_val, y_te
    """
    y_arr  = np.array(y)
    X_arr  = X.values if isinstance(X, pd.DataFrame) else X
    bins   = _k_bins(y_arr)

    # 1차: test 분리
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    tr_val_idx, te_idx = next(sss1.split(X_arr, bins))

    # 2차: val 분리 (train+val 에서)
    val_frac = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=random_state
    )
    rel_tr_idx, rel_val_idx = next(
        sss2.split(X_arr[tr_val_idx], bins[tr_val_idx])
    )
    tr_idx  = tr_val_idx[rel_tr_idx]
    val_idx = tr_val_idx[rel_val_idx]

    cols = X.columns if isinstance(X, pd.DataFrame) else range(X_arr.shape[1])
    make_X = lambda idx: pd.DataFrame(X_arr[idx], columns=cols)
    make_y = lambda idx: pd.Series(y_arr[idx], name="k_exp")

    return (make_X(tr_idx),  make_X(val_idx),  make_X(te_idx),
            make_y(tr_idx), make_y(val_idx), make_y(te_idx))


# ─────────────────────────────────────────────────────────────────────────────
# 모델 학습
# ─────────────────────────────────────────────────────────────────────────────
def _log_y(y: np.ndarray) -> np.ndarray:
    return np.log(y)

def _exp_y(y: np.ndarray) -> np.ndarray:
    return np.exp(y)


def train_single(X_tr: pd.DataFrame,
                 y_tr: pd.Series,
                 model_type: str = "gbr",
                 log_transform: bool = True
                 ) -> tuple[object, StandardScaler | None, bool]:
    """
    단일 모델 학습.

    Returns
    -------
    model, scaler (Ridge 전용, 나머지 None), log_transform_flag
    """
    y = np.array(y_tr)
    y_fit = _log_y(y) if log_transform else y
    X = X_tr.values

    scaler = None
    if model_type == "ridge":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        model = Ridge(**RIDGE_PARAMS)
    elif model_type == "rf":
        model = RandomForestRegressor(**RF_PARAMS)
    else:  # gbr (default)
        model = GradientBoostingRegressor(**GBR_PARAMS)

    model.fit(X, y_fit)
    return model, scaler, log_transform


def train_all_models(X_tr: pd.DataFrame,
                     y_tr: pd.Series,
                     log_transform: bool = True,
                     verbose: bool = True
                     ) -> dict:
    """
    GBR · RF · Ridge 3종 학습 후 dict 반환

    Returns
    -------
    {
      "gbr":   (model, scaler, log_flag),
      "rf":    (model, scaler, log_flag),
      "ridge": (model, scaler, log_flag),
    }
    """
    models = {}
    for mtype in ["gbr", "rf", "ridge"]:
        m, sc, lf = train_single(X_tr, y_tr, mtype, log_transform)
        models[mtype] = (m, sc, lf)
        if verbose:
            print(f"  [trained] {mtype.upper()}")
    return models


# ─────────────────────────────────────────────────────────────────────────────
# 예측 & 평가
# ─────────────────────────────────────────────────────────────────────────────
def predict_one(model_tuple: tuple,
                X_new: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    (model, scaler, log_flag) 튜플로 예측 → k 공간 값 반환
    """
    model, scaler, log_flag = model_tuple
    X = X_new.values if isinstance(X_new, pd.DataFrame) else X_new
    if scaler is not None:
        X = scaler.transform(X)
    y_pred = model.predict(X)
    return _exp_y(y_pred) if log_flag else y_pred


def evaluate(model_tuple: tuple,
             X: pd.DataFrame,
             y: pd.Series,
             label: str = "") -> dict:
    """
    예측 후 k-space + log-space + low-k 구간 지표 반환
    """
    y_pred = predict_one(model_tuple, X)
    y_true = np.array(y)

    # k-space 지표
    m = _metrics(y_true, y_pred)

    # log-space 지표 (RMSE 오염 방지용)
    log_m = _metrics(np.log(y_true), np.log(np.clip(y_pred, 1e-3, None)))
    m["log_RMSE"] = round(log_m["RMSE"], 4)
    m["log_R2"]   = round(log_m["R2"],   4)

    # low-k 구간 (k < 5) 지표
    mask_lk = y_true < 5.0
    if mask_lk.sum() >= 3:
        lk_m = _metrics(y_true[mask_lk], y_pred[mask_lk])
        m["lk_RMSE"] = round(lk_m["RMSE"], 4)
        m["lk_R2"]   = round(lk_m["R2"],   4)
        m["lk_n"]    = int(mask_lk.sum())

    if label:
        lk_str = f"  low-k R²={m.get('lk_R2','N/A')}" if "lk_R2" in m else ""
        print(f"  [{label}] RMSE={m['RMSE']:.3f}  logRMSE={m['log_RMSE']:.3f}"
              f"  R²={m['R2']:.4f}{lk_str}")

    return {**m, "y_pred": y_pred, "y_true": y_true}


def predict_ensemble(models: dict,
                     X_new: pd.DataFrame) -> pd.DataFrame:
    """
    3종 모델 앙상블 예측.

    Returns DataFrame with columns:
      pred_gbr, pred_rf, pred_ridge, pred_mean, pred_std
    """
    preds = {}
    for name, mt in models.items():
        preds[f"pred_{name}"] = predict_one(mt, X_new)

    df = pd.DataFrame(preds)
    vals = df[["pred_gbr", "pred_rf", "pred_ridge"]].values
    df["pred_mean"] = vals.mean(axis=1)
    df["pred_std"]  = vals.std(axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 교차검증
# ─────────────────────────────────────────────────────────────────────────────
def cross_validate(X_tr: pd.DataFrame,
                   y_tr: pd.Series,
                   model_type: str = "gbr",
                   n_folds: int = 5,
                   log_transform: bool = True,
                   verbose: bool = True
                   ) -> dict:
    """
    K-fold CV.  각 fold 의 RMSE · MAE · R² 반환.

    Returns
    -------
    {
      "rmse_folds": [...],  "mae_folds":  [...],  "r2_folds":  [...],
      "rmse_mean":  float,  "rmse_std":   float,
      "mae_mean":   float,  "r2_mean":    float,
    }
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    X_arr = X_tr.values
    y_arr = np.array(y_tr)

    rmse_list, mae_list, r2_list = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr), 1):
        Xf_tr, Xf_val = X_arr[tr_idx], X_arr[val_idx]
        yf_tr, yf_val = y_arr[tr_idx],  y_arr[val_idx]

        mt = train_single(
            pd.DataFrame(Xf_tr, columns=X_tr.columns),
            pd.Series(yf_tr),
            model_type, log_transform
        )
        yf_pred = predict_one(mt, Xf_val)
        m = _metrics(yf_val, yf_pred)

        rmse_list.append(m["RMSE"])
        mae_list.append(m["MAE"])
        r2_list.append(m["R2"])

        if verbose:
            print(f"  Fold {fold}: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}")

    result = {
        "rmse_folds": rmse_list,  "mae_folds": mae_list,  "r2_folds": r2_list,
        "rmse_mean":  float(np.mean(rmse_list)),
        "rmse_std":   float(np.std(rmse_list)),
        "mae_mean":   float(np.mean(mae_list)),
        "r2_mean":    float(np.mean(r2_list)),
        "r2_std":     float(np.std(r2_list)),
    }
    if verbose:
        print(f"\n  CV Summary ({n_folds}-fold):")
        print(f"    RMSE = {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
        print(f"    MAE  = {result['mae_mean']:.4f}")
        print(f"    R²   = {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Applicability Domain (AD) — Leverage
# ─────────────────────────────────────────────────────────────────────────────
def calc_leverage(X_tr: pd.DataFrame | np.ndarray,
                  X_new: pd.DataFrame | np.ndarray
                  ) -> tuple[np.ndarray, float]:
    """
    Williams plot 용 Leverage 점수 계산.

    h_i = x_i^T (X^T X)^{-1} x_i

    Returns
    -------
    h_new   : leverage scores for X_new (shape: n_new,)
    h_star  : warning limit = 3 * (p+1) / n_tr
    """
    Xtr = X_tr.values if isinstance(X_tr, pd.DataFrame) else X_tr
    Xnw = X_new.values if isinstance(X_new, pd.DataFrame) else X_new

    # (X^T X)^{-1} — pseudo-inverse for numerical stability
    XtX_inv = np.linalg.pinv(Xtr.T @ Xtr)

    h_new = np.array([
        float(x @ XtX_inv @ x) for x in Xnw
    ])
    n_tr, p = Xtr.shape
    h_star = 3.0 * (p + 1) / n_tr

    return h_new, h_star


# ─────────────────────────────────────────────────────────────────────────────
# 전체 파이프라인 1-call
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(X: pd.DataFrame,
                 y: pd.Series,
                 log_transform: bool = True,
                 verbose: bool = True
                 ) -> dict:
    """
    데이터 분할 → 모델 학습 → CV → 평가 → 결과 dict 반환

    Returns
    -------
    {
      "splits":  {"X_tr", "X_val", "X_te", "y_tr", "y_val", "y_te"},
      "models":  {"gbr": ..., "rf": ..., "ridge": ...},
      "cv":      {CV 결과 dict},
      "metrics": {"train": {...}, "val": {...}, "test": {...}},  # per model
    }
    """
    print("\n[ml_engine] ── 데이터 분할 ──────────────────────────")
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y)
    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}")
    print(f"  Train k: {y_tr.min():.2f}~{y_tr.max():.2f}  "
          f"Test k: {y_te.min():.2f}~{y_te.max():.2f}")

    print("\n[ml_engine] ── 모델 학습 ───────────────────────────")
    models = train_all_models(X_tr, y_tr, log_transform, verbose)

    print("\n[ml_engine] ── 5-fold CV (GBR) ────────────────────")
    cv_result = cross_validate(X_tr, y_tr, "gbr", 5, log_transform, verbose)

    print("\n[ml_engine] ── 성능 평가 ───────────────────────────")
    metrics = {}
    for mname, mt in models.items():
        print(f"\n  {mname.upper()}:")
        metrics[mname] = {
            "train": evaluate(mt, X_tr,  y_tr,  f"{mname} train"),
            "val":   evaluate(mt, X_val, y_val, f"{mname} val"),
            "test":  evaluate(mt, X_te,  y_te,  f"{mname} test"),
        }

    return {
        "splits": {
            "X_tr": X_tr, "X_val": X_val, "X_te": X_te,
            "y_tr": y_tr, "y_val": y_val, "y_te": y_te,
        },
        "models":  models,
        "cv":      cv_result,
        "metrics": metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 독립 실행 검증
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_engine import build_feature_matrix, select_features

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "crc_dielectric_full.csv"
    )
    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path)

    # 피처 행렬 구성
    print("\n[Step 1] 피처 계산...")
    X_all, y = build_feature_matrix(df, verbose=True)

    print("\n[Step 2] 피처 선택...")
    X_sel, feat_names = select_features(X_all, y, n_features=50, verbose=False)
    print(f"  선택된 피처: {len(feat_names)}개")

    # 전체 ML 파이프라인
    print("\n" + "="*55)
    result = run_pipeline(X_sel, y, log_transform=True, verbose=True)

    # GBR 최종 성능 요약
    print("\n" + "="*55)
    print("  GBR 최종 성능 요약")
    print("="*55)
    gbr_m = result["metrics"]["gbr"]
    for split, m in gbr_m.items():
        print(f"  {split:>5}: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}")

    # 앙상블 예측 테스트
    print("\n[앙상블 예측 테스트]")
    X_te  = result["splits"]["X_te"]
    y_te  = result["splits"]["y_te"]
    ens   = predict_ensemble(result["models"], X_te)
    print(ens[["pred_gbr","pred_rf","pred_ridge","pred_mean","pred_std"]].head(8).to_string(index=False))

    # AD 테스트
    print("\n[AD 레버리지 테스트]")
    X_tr = result["splits"]["X_tr"]
    h, h_star = calc_leverage(X_tr, X_te)
    n_in  = (h <= h_star).sum()
    n_out = (h >  h_star).sum()
    print(f"  h* (경고 한계) = {h_star:.4f}")
    print(f"  AD 내부: {n_in}개  |  AD 외부: {n_out}개")
