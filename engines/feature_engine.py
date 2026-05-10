# -*- coding: utf-8 -*-
"""
feature_engine.py
─────────────────
RDKit 기반 분자 기술자 계산 및 피처 선택 엔진

주요 함수
  build_feature_matrix(df)         : 전체 데이터셋 → 피처 행렬 반환
  select_features(X, y, n=50)      : 피처 선택 파이프라인 (분산→상관→중요도)
  calc_single(smiles)              : 단일 SMILES → 기술자 dict 반환

디자인 원칙
  - RDKit 내장 기술자만 사용 (Mordred 불필요)
  - 수치 불안정 기술자(Ipc, AvgIpc) 제외
  - F·Si·N·O·S 원자 count 커스텀 기술자 추가
  - 결측값 중앙값 대체 후 반환
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 수치 불안정 기술자 제외 목록
# ─────────────────────────────────────────────────────────────────────────────
_EXCLUDE = {"Ipc", "AvgIpc", "SPS", "qed"}

# RDKit 기술자 함수 목록 (이름, 함수) — 제외 목록 적용
_RDKIT_DESCS: list[tuple[str, object]] = [
    (name, fn)
    for name, fn in Descriptors.descList
    if name not in _EXCLUDE
]


# ─────────────────────────────────────────────────────────────────────────────
# 커스텀 기술자 (원자 카운트 기반)
# ─────────────────────────────────────────────────────────────────────────────
def _custom_descriptors(mol) -> dict:
    """
    F, Cl, Br, Si, N, O, S 원자 수 및 분율 계산
    유전율과 직접 관련된 원소 기반 기술자
    """
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    n_total = len(atoms) if atoms else 1

    counts = {
        "n_F":  atoms.count(9),
        "n_Cl": atoms.count(17),
        "n_Br": atoms.count(35),
        "n_Si": atoms.count(14),
        "n_N":  atoms.count(7),
        "n_O":  atoms.count(8),
        "n_S":  atoms.count(16),
        "n_C":  atoms.count(6),
    }

    fracs = {
        f"frac_{k[2:]}": v / n_total
        for k, v in counts.items()
    }

    # 할로겐 합계
    counts["n_halogen"] = counts["n_F"] + counts["n_Cl"] + counts["n_Br"]
    fracs["frac_halogen"] = counts["n_halogen"] / n_total

    # 불소화도 (fluorination degree): F 수 / (C 수 + F 수), 탄화불소 특성
    c_plus_f = counts["n_C"] + counts["n_F"]
    fracs["fluoro_degree"] = counts["n_F"] / c_plus_f if c_plus_f > 0 else 0.0

    # Si-O 결합 수 (실록산 특성)
    si_o_count = sum(
        1
        for bond in mol.GetBonds()
        if {bond.GetBeginAtom().GetAtomicNum(),
            bond.GetEndAtom().GetAtomicNum()} == {14, 8}
    )
    counts["n_SiO_bonds"] = si_o_count

    return {**counts, **fracs}


def _calc_one(smiles: str) -> dict | None:
    """
    단일 SMILES → 모든 기술자(RDKit + 커스텀) dict 반환
    실패 시 None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    row: dict = {}

    # RDKit 내장 기술자 (~213개)
    for name, fn in _RDKIT_DESCS:
        try:
            val = fn(mol)
            row[name] = float(val) if val is not None else np.nan
        except Exception:
            row[name] = np.nan

    # 커스텀 기술자 (~16개)
    try:
        row.update(_custom_descriptors(mol))
    except Exception:
        pass

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def calc_single(smiles: str) -> dict | None:
    """
    단일 SMILES → 기술자 dict 반환  (스크리닝 탭에서 사용)
    """
    return _calc_one(smiles)


def build_feature_matrix(df: pd.DataFrame,
                          smiles_col: str = "smiles",
                          verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    데이터셋 전체 → (X: 피처 DataFrame, y: k_exp Series)

    Parameters
    ----------
    df         : crc_dielectric_full.csv 를 읽어 넘긴 DataFrame
    smiles_col : SMILES 컬럼명
    verbose    : 진행 상황 출력 여부

    Returns
    -------
    X : shape (n_valid, n_features)  — NaN 결측값 없음 (중앙값 대체)
    y : shape (n_valid,)             — k_exp 값
    """
    rows = []
    valid_idx = []

    for i, row in df.iterrows():
        desc = _calc_one(row[smiles_col])
        if desc is not None:
            rows.append(desc)
            valid_idx.append(i)

    if verbose:
        print(f"[feature_engine] 계산 완료: {len(rows)}/{len(df)} 분자")

    X = pd.DataFrame(rows, index=valid_idx)
    y = df.loc[valid_idx, "k_exp"].reset_index(drop=True)
    X = X.reset_index(drop=True)

    # 결측값 → 컬럼별 중앙값 대체
    medians = X.median()
    X = X.fillna(medians)

    # 여전히 NaN 인 컬럼(전체 NaN) 제거
    X = X.dropna(axis=1)

    if verbose:
        print(f"[feature_engine] 피처 수: {X.shape[1]}  |  분자 수: {X.shape[0]}")

    return X, y


def select_features(X: pd.DataFrame,
                    y: pd.Series,
                    n_features: int = 50,
                    corr_threshold: float = 0.95,
                    verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    3단계 피처 선택 파이프라인

    Step 1. 분산 0 제거  (VarianceThreshold)
    Step 2. 상관계수 >corr_threshold 중 하나 제거
    Step 3. RandomForest 중요도 기반 상위 n_features 선택

    Returns
    -------
    X_sel      : 선택된 피처만 포함한 DataFrame
    feat_names : 선택된 피처 이름 리스트
    """
    # ── Step 1: 분산 0 제거 ───────────────────────────────────────────────────
    sel_var = VarianceThreshold(threshold=0.0)
    X_arr = sel_var.fit_transform(X)
    cols_after_var = X.columns[sel_var.get_support()].tolist()
    X1 = pd.DataFrame(X_arr, columns=cols_after_var)
    if verbose:
        print(f"[select] Step1 분산 필터: {X.shape[1]} → {X1.shape[1]} 피처")

    # ── Step 2: 높은 상관 제거 ───────────────────────────────────────────────
    corr_matrix = X1.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    X2 = X1.drop(columns=to_drop)
    if verbose:
        print(f"[select] Step2 상관 필터 (>{corr_threshold}): "
              f"{X1.shape[1]} → {X2.shape[1]} 피처")

    # ── Step 3: RF 중요도 기반 상위 N 선택 ──────────────────────────────────
    n_select = min(n_features, X2.shape[1])
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X2, y)

    importances = pd.Series(rf.feature_importances_, index=X2.columns)
    top_feats = importances.nlargest(n_select).index.tolist()
    X_sel = X2[top_feats].copy()

    if verbose:
        print(f"[select] Step3 RF 중요도: {X2.shape[1]} → {X_sel.shape[1]} 피처")
        print(f"\n  Top-10 피처:")
        for rank, (feat, imp) in enumerate(
                importances[top_feats].items(), 1):
            bar = "#" * int(imp * 200)
            print(f"  {rank:>2}. {feat:<30} {imp:.4f}  {bar}")

    return X_sel, top_feats


def get_feature_importances(X: pd.DataFrame,
                             y: pd.Series) -> pd.Series:
    """RF 기반 전체 피처 중요도 반환 (시각화용)"""
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 독립 실행 검증
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "crc_dielectric_full.csv"
    )
    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset: {len(df)} molecules\n")

    # 1. 전체 피처 행렬 구성
    X_all, y = build_feature_matrix(df, verbose=True)
    print(f"\nX_all shape: {X_all.shape}")
    print(f"y range: {y.min():.2f} ~ {y.max():.2f}")

    # 2. 피처 선택
    print("\n" + "="*50)
    X_sel, feat_names = select_features(X_all, y, n_features=50, verbose=True)
    print(f"\nFinal feature matrix: {X_sel.shape}")

    # 3. 단일 분자 테스트
    print("\n" + "="*50)
    print("Single molecule test (Hexane, k=1.886):")
    desc = calc_single("CCCCCC")
    key_feats = ["MolWt", "MolLogP", "MolMR", "TPSA", "FractionCSP3",
                 "NumAromaticRings", "n_F", "n_Si", "frac_F", "fluoro_degree"]
    for f in key_feats:
        val = desc.get(f, "N/A")
        print(f"  {f:<25}: {val}")

    # 4. 결과 저장 (검증용)
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "features_all.csv"
    )
    X_all_with_y = X_all.copy()
    X_all_with_y.insert(0, "k_exp", y)
    X_all_with_y.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[saved] {out_path}")
