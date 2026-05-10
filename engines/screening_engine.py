# -*- coding: utf-8 -*-
"""
screening_engine.py
───────────────────
후보 분자 스크리닝 엔진

주요 함수
  screen(smiles_input, feat_names, models, X_tr) : 메인 스크리닝
  draw_molecules(smiles_list, preds)              : 구조 이미지 그리드 반환
  load_example_candidates()                       : 내장 후보 목록

입력 형식 (smiles_input)
  - str  : "Name,SMILES\nName2,SMILES2\n..."  (이름,SMILES 한 줄씩)
  - list : [(name, smiles), ...]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import io
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 내장 후보 목록  (문헌 기반 low-k 후보 SMILES)
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLE_CANDIDATES = [
    # Fluorinated aromatics / low-k specials
    ("Hexafluorobenzene",          "Fc1c(F)c(F)c(F)c(F)c1F"),
    ("Octafluoronaphthalene",      "Fc1c(F)c(F)c2c(F)c(F)c(F)c(F)c2c1F"),
    ("Perfluorotoluene",           "FC(F)(F)c1c(F)c(F)c(F)c(F)c1F"),
    ("1,4-Difluorobenzene",        "Fc1ccc(F)cc1"),
    ("Pentafluorobenzene",         "Fc1c(F)c(F)c(F)c(F)c1"),
    # Siloxanes
    ("Hexamethyldisiloxane",       "C[Si](C)(C)O[Si](C)(C)C"),
    ("Octamethylcyclotetrasiloxane","C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1"),
    ("Decamethylcyclopentasiloxane","C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1"),
    ("HMDS (trimethylsilylamine)",  "C[Si](C)(C)N[Si](C)(C)C"),
    ("Tetramethylsilane",          "C[Si](C)(C)C"),
    # Fluorinated alkanes / ethers
    ("Perfluoropentane",           "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),
    ("Perfluorohexane",            "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),
    ("Perfluorocyclobutane",       "FC1(F)C(F)(F)C(F)(F)C1(F)F"),
    ("PFPE-like ether",            "FC(F)(F)OC(F)(F)OC(F)(F)F"),
    # Aromatic low-k
    ("Biphenyl",                   "c1ccc(-c2ccccc2)cc1"),
    ("p-Terphenyl",                "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"),
    ("Naphthalene",                "c1ccc2ccccc2c1"),
    ("Anthracene",                 "c1ccc2cc3ccccc3cc2c1"),
    ("Pyrene",                     "c1ccc2ccc3cccc4ccc1c2c34"),
    ("Fluorene",                   "C1c2ccccc2-c2ccccc21"),
    # BCB-type / low-k special
    ("Benzocyclobutene (BCB)",     "C1CC2=CC=CC=C2C1=C"),
    ("p-Diethynylbenzene",         "C#Cc1ccc(C#C)cc1"),
    ("1,3,5-Triethynylbenzene",    "C#Cc1cc(C#C)cc(C#C)c1"),
    # Polyimide precursors (low polarity variants)
    ("4,4'-Diaminodiphenyl ether", "Nc1ccc(Oc2ccc(N)cc2)cc1"),
    ("Hexafluoro-BPDA dianhydride","O=C1OC(=O)c2cc(C(F)(F)F)cc3c2c1ccc3C(F)(F)F"),
    # Aliphatic / cyclic low-k
    ("Adamantane",                 "C1C2CC3CC1CC(C2)C3"),
    ("Norbornane",                 "C1CC2CC1CC2"),
    ("Diamondoid (Diamantane)",    "C1C2CC3CC1CC(C2)C3"),
    ("Decalin",                    "C1CCC2CCCCC2C1"),
    ("trans-Bicyclo[2.2.0]hexane", "C1CCC2CCC1C2"),
]


def load_example_candidates() -> list[tuple[str, str]]:
    """유효한 SMILES만 필터링해서 반환"""
    valid = []
    for name, smi in EXAMPLE_CANDIDATES:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid.append((name, smi))
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# 메인 스크리닝
# ─────────────────────────────────────────────────────────────────────────────
def screen(
    smiles_input,
    feat_names: list[str],
    models: dict,
    X_tr: pd.DataFrame,
) -> pd.DataFrame:
    """
    후보 분자 스크리닝.

    Parameters
    ----------
    smiles_input : str (Name,SMILES\\n...) 또는 list of (name, smiles)
    feat_names   : 훈련에 사용된 피처 이름 리스트
    models       : {"gbr": (...), "rf": (...), "ridge": (...)}
    X_tr         : 훈련 데이터 피처 행렬 (AD 계산용)

    Returns
    -------
    DataFrame with columns:
      rank, name, smiles, valid,
      pred_gbr, pred_rf, pred_ridge, pred_mean, pred_std,
      h, ad_ok, ad_label
    """
    # ── 입력 파싱 ─────────────────────────────────────────────────────────────
    if isinstance(smiles_input, str):
        pairs = _parse_text(smiles_input)
    else:
        pairs = list(smiles_input)

    if not pairs:
        return pd.DataFrame()

    # ── 기술자 계산 ───────────────────────────────────────────────────────────
    from engines.feature_engine import calc_single

    rows_meta   = []
    rows_feats  = []

    for name, smi in pairs:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows_meta.append({"name": name, "smiles": smi, "valid": False})
            rows_feats.append(None)
            continue

        desc = calc_single(smi)
        if desc is None:
            rows_meta.append({"name": name, "smiles": smi, "valid": False})
            rows_feats.append(None)
        else:
            rows_meta.append({"name": name, "smiles": smi, "valid": True})
            rows_feats.append(desc)

    # ── 유효 분자만 예측 ──────────────────────────────────────────────────────
    valid_idx = [i for i, r in enumerate(rows_feats) if r is not None]
    if not valid_idx:
        return _build_invalid_df(rows_meta)

    feat_df = pd.DataFrame(
        [rows_feats[i] for i in valid_idx]
    )

    # 훈련 피처셋으로 정렬 (없는 피처 = 0으로 채움)
    for col in feat_names:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    X_new = feat_df[feat_names].fillna(0.0)

    # ── 앙상블 예측 ───────────────────────────────────────────────────────────
    from engines.ml_engine import predict_one, calc_leverage

    preds = {}
    for mname, mt in models.items():
        preds[f"pred_{mname}"] = predict_one(mt, X_new)

    pred_vals = np.column_stack(list(preds.values()))
    pred_mean = pred_vals.mean(axis=1)
    pred_std  = pred_vals.std(axis=1)

    # ── Leverage (AD) ─────────────────────────────────────────────────────────
    h_new, h_star = calc_leverage(X_tr[feat_names], X_new)
    ad_ok    = h_new <= h_star
    ad_label = np.where(ad_ok, "AD 내부 ✅", "AD 외부 ⚠️")

    # ── 결과 조합 ─────────────────────────────────────────────────────────────
    result_rows = []
    valid_ptr = 0
    for i, meta in enumerate(rows_meta):
        if not meta["valid"]:
            result_rows.append({
                **meta,
                "pred_gbr":   None, "pred_rf":    None, "pred_ridge": None,
                "pred_mean":  None, "pred_std":   None,
                "h":          None, "ad_ok":      None, "ad_label":   "유효하지 않은 SMILES ❌",
            })
        else:
            vp = valid_ptr
            result_rows.append({
                **meta,
                "pred_gbr":   round(float(preds.get("pred_gbr",  [None]*100)[vp] or 0), 3),
                "pred_rf":    round(float(preds.get("pred_rf",   [None]*100)[vp] or 0), 3),
                "pred_ridge": round(float(preds.get("pred_ridge",[None]*100)[vp] or 0), 3),
                "pred_mean":  round(float(pred_mean[vp]), 3),
                "pred_std":   round(float(pred_std[vp]),  3),
                "h":          round(float(h_new[vp]),     4),
                "ad_ok":      bool(ad_ok[vp]),
                "ad_label":   ad_label[vp],
            })
            valid_ptr += 1

    df_result = pd.DataFrame(result_rows)

    # 유효 분자만 pred_mean 기준 정렬
    valid_mask = df_result["valid"] == True
    df_valid   = df_result[valid_mask].sort_values("pred_mean").reset_index(drop=True)
    df_invalid = df_result[~valid_mask]
    df_out     = pd.concat([df_valid, df_invalid], ignore_index=True)
    df_out.insert(0, "rank", range(1, len(df_out) + 1))

    return df_out


def _parse_text(text: str) -> list[tuple[str, str]]:
    """Name,SMILES 형식 텍스트 파싱 (SMILES만 있으면 자동 이름 부여)"""
    pairs = []
    for i, line in enumerate(text.strip().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            parts = line.split(",", 1)
            pairs.append((parts[0].strip(), parts[1].strip()))
        else:
            pairs.append((f"Mol_{i:03d}", line))
    return pairs


def _build_invalid_df(meta_rows: list) -> pd.DataFrame:
    for r in meta_rows:
        r.update({
            "pred_mean": None, "pred_std": None,
            "h": None, "ad_ok": None, "ad_label": "유효하지 않은 SMILES ❌",
        })
    df = pd.DataFrame(meta_rows)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 구조 이미지 생성
# ─────────────────────────────────────────────────────────────────────────────
def mol_to_png_bytes(smiles: str, size: int = 300) -> bytes | None:
    """단일 SMILES → PNG bytes (Streamlit st.image 용)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        import io as _io
        img = Draw.MolToImage(mol, size=(size, size))
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def draw_grid(smiles_list: list[str],
              names: list[str],
              preds: list[float],
              n_cols: int = 4,
              mol_size: tuple = (220, 180)) -> bytes | None:
    """여러 분자 그리드 이미지 → PNG bytes"""
    mols, legends = [], []
    for smi, name, pred in zip(smiles_list, names, preds):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            p_str = f"{pred:.3f}" if pred is not None else "?"
            legends.append(f"{name[:18]}\nk={p_str}")

    if not mols:
        return None

    n_rows = (len(mols) + n_cols - 1) // n_cols
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=mol_size,
        legends=legends,
        returnPNG=True,
    )
    return img
