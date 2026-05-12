# -*- coding: utf-8 -*-
"""
Low-k Dielectric QSPR Platform  —  app.py
==========================================
Tab S1: Dataset EDA
Tab S2: Feature Engineering
Tab S3: ML Model Training
Tab S4: Candidate Screening
Tab S5: Paper Draft              (추후)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from rdkit import Chem

warnings.filterwarnings("ignore")

# ── 엔진 경로 등록 ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from engines.feature_engine import build_feature_matrix, select_features, get_feature_importances
from engines.ml_engine import (
    split_data, train_all_models, cross_validate, auto_select_best,
    evaluate, predict_ensemble, calc_leverage, run_pipeline
)
from engines.screening_engine import (
    screen, load_example_candidates, draw_grid, mol_to_png_bytes
)
from engines.paper_engine import generate_paper

# ── 상수 ──────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "crc_dielectric_combined.csv")
N_FEATURES = 50

# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def load_data() -> pd.DataFrame:
    if "df" not in st.session_state:
        df = pd.read_csv(DATA_PATH)
        st.session_state["df"] = df
    return st.session_state["df"]


def _autosave_paper(paper: dict, sess: dict) -> dict:
    """
    논문 초안 및 관련 결과를 output/ 폴더에 자동 저장.
    Returns dict {label: saved_path}
    """
    from datetime import date
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    saved = {}

    # 1. 논문 Markdown
    md_path = os.path.join(OUTPUT_DIR, f"lowk_qspr_paper_{today}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(paper["full_md"])
    saved["논문 Markdown"] = md_path

    # 2. 스크리닝 결과 CSV
    if "s4_result" in sess:
        sc_path = os.path.join(OUTPUT_DIR, f"screening_results_{today}.csv")
        sess["s4_result"].to_csv(sc_path, index=False, encoding="utf-8-sig")
        saved["스크리닝 결과"] = sc_path

    # 3. 핵심 수치 CSV
    res    = sess.get("ml_result", {})
    cv     = res.get("cv", {})
    gbr_te = res.get("metrics", {}).get("gbr", {}).get("test", {})
    df_sc  = sess.get("s4_result", pd.DataFrame())
    df_v   = df_sc[df_sc["valid"] == True] if "valid" in df_sc.columns else df_sc

    summary = pd.DataFrame({
        "항목": ["데이터셋 분자수", "피처 수", "GBR Test R²", "GBR Test RMSE",
                  "GBR Test logRMSE", "CV R²", "스크리닝 후보수", "k<3 예측수"],
        "값":   [
            len(sess.get("df", pd.DataFrame())),
            len(sess.get("feat_names", [])),
            round(gbr_te.get("R2",   0), 4),
            round(gbr_te.get("RMSE", 0), 4),
            gbr_te.get("log_RMSE", "–"),
            f"{cv.get('r2_mean',0):.3f}±{cv.get('r2_std',0):.3f}",
            len(df_v),
            int((df_v["pred_mean"] < 2.4).sum()) if "pred_mean" in df_v.columns else "–",
        ],
    })
    km_path = os.path.join(OUTPUT_DIR, f"key_metrics_{today}.csv")
    summary.to_csv(km_path, index=False, encoding="utf-8-sig")
    saved["핵심 수치"] = km_path

    return saved


def _fig_to_st(fig, caption: str = ""):
    st.pyplot(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# S1 — Dataset EDA
# ─────────────────────────────────────────────────────────────────────────────
def render_s1():
    st.subheader("📊 Dataset EDA")
    df = load_data()

    # ── 기본 통계 카드 ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total molecules",  f"{len(df)}")
    c2.metric("k range",          f"{df['k_exp'].min():.2f} ~ {df['k_exp'].max():.2f}")
    c3.metric("k < 2.4 (Low-k)", f"{(df['k_exp']<2.4).sum()}")
    c4.metric("k < 2.0 (Ultra)",  f"{(df['k_exp']<2.0).sum()}")

    st.markdown("---")

    # ── 그래프 1·2 나란히 ─────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**k Distribution (full range)**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df["k_exp"], bins=30, color="#3498db", edgecolor="white", alpha=0.85)
        ax.axvline(2.4, color="red",    ls="--", lw=1.5, label="k = 2.4")
        ax.axvline(2.0, color="orange", ls="--", lw=1.5, label="k = 2.0")
        ax.set_xlabel("Dielectric constant k")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of k_exp")
        ax.legend(fontsize=8)
        plt.tight_layout()
        _fig_to_st(fig)

    with col_r:
        st.markdown("**k Distribution (low-k zoom: k < 6)**")
        df_zoom = df[df["k_exp"] < 6]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df_zoom["k_exp"], bins=25, color="#27ae60", edgecolor="white", alpha=0.85)
        ax.axvline(2.4, color="red",    ls="--", lw=1.5, label="k = 2.4")
        ax.axvline(2.0, color="orange", ls="--", lw=1.5, label="k = 2.0")
        ax.set_xlabel("Dielectric constant k")
        ax.set_ylabel("Count")
        ax.set_title(f"Low-k zoom  (n={len(df_zoom)})")
        ax.legend(fontsize=8)
        plt.tight_layout()
        _fig_to_st(fig)

    # ── 그래프 3: Category boxplot ────────────────────────────────────────────
    st.markdown("**k by Category**")
    cats = df.groupby("category")["k_exp"].median().sort_values()
    ordered_cats = cats.index.tolist()
    data_by_cat = [df[df["category"] == c]["k_exp"].values for c in ordered_cats]

    fig, ax = plt.subplots(figsize=(12, 4))
    bp = ax.boxplot(data_by_cat, patch_artist=True, vert=True,
                    medianprops=dict(color="black", lw=2))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(ordered_cats)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(ordered_cats) + 1))
    ax.set_xticklabels(ordered_cats, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("k_exp")
    ax.set_title("Dielectric constant by chemical category")
    ax.axhline(2.4, color="red", ls="--", lw=1, label="k = 2.4")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _fig_to_st(fig, "낮을수록 Low-k 후보 (green → red 순)")

    # ── Low-k 후보 테이블 ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Low-k Candidates  (k < 2.4)**")
    lowk = (df[df["k_exp"] < 2.4]
            .sort_values("k_exp")[["name","k_exp","category","source"]]
            .reset_index(drop=True))
    lowk.index += 1
    st.dataframe(lowk, use_container_width=True, height=350)

    # ── 소스별 카운트 ──────────────────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Source breakdown**")
        st.dataframe(df["source"].value_counts().rename("count"), use_container_width=True)
    with col_b:
        st.markdown("**Category counts (sorted by median k)**")
        cat_summary = (df.groupby("category")["k_exp"]
                         .agg(["count","min","max","median"])
                         .sort_values("median")
                         .round(2))
        st.dataframe(cat_summary, use_container_width=True)

    # ── 논문 출처 패널 ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📚 데이터 출처 & 참고 논문")

    tab_src1, tab_src2, tab_src3 = st.tabs(
        ["📖 출처별 분포", "🔬 논문 목록 (Literature)", "➕ 데이터 추가"]
    )

    with tab_src1:
        src_counts = df["source"].value_counts().reset_index()
        src_counts.columns = ["source", "count"]
        c_left, c_right = st.columns([2, 3])
        with c_left:
            st.dataframe(src_counts, use_container_width=True, hide_index=True)
        with c_right:
            fig_s, ax_s = plt.subplots(figsize=(5, 3))
            top_src = src_counts.head(8)
            ax_s.barh(top_src["source"].str[:30][::-1],
                      top_src["count"][::-1],
                      color="#3498db", edgecolor="white", alpha=0.85)
            ax_s.set_xlabel("Count")
            ax_s.set_title("Data sources")
            plt.tight_layout()
            _fig_to_st(fig_s)

    with tab_src2:
        st.markdown("**논문 기반 데이터셋 (DOI 포함)**")
        PAPER_LIST = [
            ("BCB / Cyclotene", "Microelectron. Eng. 1996",
             "10.1016/S0167931796000615", "k=2.65 @ 1 MHz; DVS-BCB fully cured"),
            ("BCB / Cyclotene (GHz)", "J. Infrared Millim. Terahertz Waves 2009",
             "10.1007/s10762-009-9552-0", "k=2.49 @ 11-65 GHz"),
            ("Parylene AF4", "J. Fluorine Chem. 2003",
             "10.1016/S0022-1139(03)00100-3", "k=2.3; CVD; best thermal stability"),
            ("Parylene HT", "IEEE ECTC 2023",
             "10.1109/ECTC51909.2023.00177", "k=2.17 @ 1 MHz"),
            ("SiLK polyphenylene", "MRS Proc. 1997",
             "10.1557/PROC-476-9", "k=2.65; spin-on crosslinked"),
            ("Teflon AF (porous)", "Mater. Sci. Eng. B 2001",
             "10.1016/S0921-5107(01)00504-9", "k=1.57 @ 1 MHz; porous"),
            ("MSQ dense/porous", "Mol. Cryst. Liq. Cryst. 2001",
             "10.1080/10587250108024702", "k=2.7→2.2 with 30% porosity"),
            ("HSQ dense", "Thin Solid Films 1998",
             "10.1016/S0040-6090(98)01041-0", "k=2.85; cured 400°C"),
            ("Porous HSQ", "J. Am. Ceram. Soc. 2007",
             "10.1111/j.1551-2916.2007.01713.x", "k=2.2; D4 porogen"),
            ("BCB-POSS T8/T10", "Polymer 2023",
             "10.1016/j.polymer.2023.126189", "k=2.24/2.02 @ 1 kHz; cage size effect"),
            ("F-POSS in PI", "Polym. Eng. Sci. 2022",
             "10.1002/pen.26063", "k=2.47; trifluoropropyl POSS 2wt%"),
            ("6FDA-TFMB PI", "ACS Appl. Polym. Mater. 2021",
             "10.1021/acsapm.0c01141", "k=2.72 @ 10 GHz; Df=0.0075"),
            ("6FDA-TFMB ultra-low", "Eur. Polym. J. 2021",
             "10.1016/j.eurpolymj.2020.109904", "k=1.84; branched crosslink"),
            ("F-polynorbornene", "ACS Appl. Polym. Mater. 2020",
             "10.1021/acsapm.9b01070", "k=2.39 @ 5 GHz; cross-linked"),
            ("Nanoporous PMSSQ", "Nature Materials 2005",
             "10.1038/nmat1291", "k~1.5; dendrimer imprint; <2nm pores"),
            ("── Review Papers ──", "", "", ""),
            ("Low-k polymers review", "Prog. Polym. Sci. 2001 (Maier)",
             "10.1016/S0079-6700(00)00043-5", "248 refs; PI, BCB, parylene, SiLK"),
            ("Low-k materials review", "Chem. Rev. 2010 (Volksen et al.)",
             "10.1021/cr9002819", "Organosilicate, porous ULK focus"),
        ]
        rows_paper = []
        for name, journal, doi, note in PAPER_LIST:
            doi_link = f"[{doi}](https://doi.org/{doi})" if doi else "–"
            rows_paper.append({"재료": name, "저널/출처": journal,
                                "DOI": doi, "주요 k값": note})
        df_papers = pd.DataFrame(rows_paper)
        st.dataframe(df_papers, use_container_width=True, hide_index=True)

    with tab_src3:
        st.markdown("**직접 데이터 추가 (논문 k값 → 데이터셋)**")
        st.info("새 분자를 추가하면 다음 S2 피처 계산 시 자동 반영됩니다.")
        col_add1, col_add2 = st.columns(2)
        with col_add1:
            new_smiles = st.text_input("SMILES", placeholder="C[Si](C)(C)O[Si](C)(C)C")
            new_name   = st.text_input("화합물명", placeholder="Hexamethyldisiloxane")
            new_k      = st.number_input("k 값 (실험값)", 1.0, 10.0, 2.5, 0.01)
        with col_add2:
            new_cat    = st.selectbox("카테고리",
                         ["silicon","fluorinated","aromatic","bcb_parylene",
                          "low_k_special","polymer","alkane","ether","other"])
            new_src    = st.text_input("출처 (저널/데이터시트)", placeholder="J. Mater. Chem. 2023")
            new_doi    = st.text_input("DOI (선택)", placeholder="10.1039/...")

        if st.button("➕ 데이터셋에 추가"):
            mol = Chem.MolFromSmiles(new_smiles) if new_smiles else None
            if mol is None:
                st.error("유효하지 않은 SMILES입니다.")
            elif not new_name:
                st.error("화합물명을 입력하세요.")
            else:
                new_row = pd.DataFrame([{
                    "smiles": new_smiles, "k_exp": new_k,
                    "name": new_name, "category": new_cat,
                    "source": new_src, "doi": new_doi,
                    "note": "user added", "rdkit_valid": True,
                    "smiles_src": "user",
                }])
                # 기존 df에 추가 & 저장
                df_updated = pd.concat([df, new_row], ignore_index=True)
                df_updated.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
                st.session_state["df"] = df_updated
                st.success(f"✅ '{new_name}' (k={new_k}) 추가 완료! "
                           f"총 {len(df_updated)}개. S2 탭에서 재계산하세요.")

    # S1 완료 플래그
    st.session_state["s1_done"] = True


# ─────────────────────────────────────────────────────────────────────────────
# S2 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def render_s2():
    st.subheader("🔬 Feature Engineering")

    df = load_data()

    # 캐시 확인
    if "X_sel" in st.session_state:
        st.success(f"✅ 피처 행렬 캐시됨  —  {st.session_state['X_sel'].shape[1]}개 선택 피처 / "
                   f"{st.session_state['X_all'].shape[0]}개 분자")
        rerun = st.button("🔄 피처 재계산")
        if not rerun:
            _render_s2_plots()
            return
        else:
            for k in ["X_all","y","X_sel","feat_names","feat_imp"]:
                st.session_state.pop(k, None)

    st.info(f"RDKit 기술자 계산 중...  ({len(df)}개 분자)")
    prog = st.progress(0, text="계산 중...")

    # ── 피처 계산 ─────────────────────────────────────────────────────────────
    with st.spinner("분자 기술자 계산 중..."):
        X_all, y = build_feature_matrix(df, verbose=False)
        prog.progress(40, text="피처 선택 중...")

    with st.spinner("피처 선택 중..."):
        X_sel, feat_names = select_features(X_all, y, n_features=N_FEATURES, verbose=False)
        prog.progress(80, text="중요도 계산 중...")

    with st.spinner("피처 중요도 계산 중..."):
        feat_imp = get_feature_importances(X_sel, y)
        prog.progress(100, text="완료!")

    # session_state 저장
    st.session_state["X_all"]      = X_all
    st.session_state["y"]          = y
    st.session_state["X_sel"]      = X_sel
    st.session_state["feat_names"] = feat_names
    st.session_state["feat_imp"]   = feat_imp

    prog.empty()
    st.success(f"✅ 피처 계산 완료  —  "
               f"{X_all.shape[1]}개 원시 → {X_sel.shape[1]}개 선택")

    _render_s2_plots()


def _render_s2_plots():
    X_all      = st.session_state["X_all"]
    y          = st.session_state["y"]
    X_sel      = st.session_state["X_sel"]
    feat_imp   = st.session_state["feat_imp"]

    st.markdown("---")

    # ── 파이프라인 요약 ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw features",        f"{X_all.shape[1]}")
    c2.metric("After variance + corr filter", f"{X_all.shape[1]} → ~{X_sel.shape[1]+20}")
    c3.metric("Final selected",      f"{X_sel.shape[1]}")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        # ── 피처 중요도 바 차트 ───────────────────────────────────────────────
        st.markdown("**Top-20 Feature Importances  (RF-based)**")
        top20 = feat_imp.head(20)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top20)))[::-1]
        bars = ax.barh(range(len(top20)), top20.values[::-1],
                       color=colors, edgecolor="white")
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20.index[::-1], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title("RF Feature Importance (Top 20)")
        plt.tight_layout()
        _fig_to_st(fig)

    with col_r:
        # ── 상위 10 피처 설명 테이블 ─────────────────────────────────────────
        st.markdown("**Top-10 Features**")
        desc_map = {
            "MolLogP":             "소수성 (k ↓ 연관)",
            "TPSA":                "극성 표면적 (k ↑ 연관)",
            "MolMR":               "몰 굴절률 (분극률)",
            "MolWt":               "분자량",
            "FractionCSP3":        "sp3 탄소 비율",
            "NumAromaticRings":    "방향족 고리 수",
            "NumHDonors":          "H-bond 주개 수",
            "NumHAcceptors":       "H-bond 받개 수",
            "BCUT2D_LOGPHI":       "logP 기반 연결성",
            "MinAbsPartialCharge": "최소 부분 전하",
            "MaxAbsPartialCharge": "최대 부분 전하",
            "n_F":                 "불소 원자 수",
            "frac_F":              "불소 원자 분율",
            "n_Si":                "Si 원자 수",
            "fluoro_degree":       "불소화도",
            "TPSA":                "극성 표면적",
        }
        top10_df = pd.DataFrame({
            "Feature":     feat_imp.head(10).index,
            "Importance":  feat_imp.head(10).values.round(4),
            "Meaning":     [desc_map.get(f, "-") for f in feat_imp.head(10).index],
        })
        st.dataframe(top10_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**3-Step Feature Selection**")
        st.markdown("""
| Step | Method | 결과 |
|------|--------|------|
| 1 | Variance = 0 제거 | ~45개 제거 |
| 2 | Corr > 0.95 제거 | ~46개 제거 |
| 3 | RF 중요도 Top-50 | **50개 확정** |
""")

    # ── k vs MolLogP 산점도 ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Key Feature vs k  (MolLogP · TPSA · MolMR)**")
    df = load_data()

    # MolLogP, TPSA, MolMR 을 X_all에서 가져오기
    plot_feats = ["MolLogP", "TPSA", "MolMR"]
    available  = [f for f in plot_feats if f in X_all.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 3.8))
    if len(available) == 1:
        axes = [axes]
    for ax, feat in zip(axes, available):
        colors_sc = np.where(y < 2.4, "#27ae60", np.where(y < 10.0, "#f39c12", "#e74c3c"))
        ax.scatter(X_all[feat], y, c=colors_sc, s=20, alpha=0.7, edgecolors="none")
        ax.set_xlabel(feat)
        ax.set_ylabel("k_exp")
        ax.set_title(f"{feat} vs k")
        ax.set_yscale("log")
        # 범례
        patches = [
            mpatches.Patch(color="#27ae60", label="k < 2.4"),
            mpatches.Patch(color="#f39c12", label="2.4 ≤ k < 10"),
            mpatches.Patch(color="#e74c3c", label="k ≥ 10"),
        ]
        ax.legend(handles=patches, fontsize=7)
    plt.tight_layout()
    _fig_to_st(fig, "y축 log scale. 초록=low-k 후보")

    st.session_state["s2_done"] = True


# ─────────────────────────────────────────────────────────────────────────────
# S3 — ML Model Training
# ─────────────────────────────────────────────────────────────────────────────
def render_s3():
    st.subheader("🤖 ML Model Training")

    if "X_sel" not in st.session_state:
        st.warning("⚠️ S2 탭에서 피처를 먼저 계산하세요.")
        return

    X_sel = st.session_state["X_sel"]
    y     = st.session_state["y"]

    # 캐시 확인
    if "ml_result" in st.session_state:
        st.success("✅ 모델 학습 결과 캐시됨")
        retrain = st.button("🔄 재학습")
        if not retrain:
            _render_s3_results()
            return
        else:
            st.session_state.pop("ml_result", None)

    # ── 학습 파라미터 설정 ────────────────────────────────────────────────────
    st.markdown("**학습 설정**")
    col1, col2, col3 = st.columns(3)
    test_size = col1.slider("Test size",  0.10, 0.25, 0.15, 0.05)
    val_size  = col2.slider("Val size",   0.10, 0.25, 0.15, 0.05)
    log_tf    = col3.checkbox("Log(k) 변환 학습", value=True)

    if st.button("▶ 모델 학습 시작", type="primary"):
        prog = st.progress(0, text="데이터 분할 중...")

        with st.spinner("학습 중... (약 10-20초)"):
            from engines.ml_engine import split_data, train_all_models, cross_validate, evaluate, auto_select_best
            X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(
                X_sel, y, test_size=test_size, val_size=val_size
            )
            prog.progress(15, text="GBR 학습 중...")
            models = train_all_models(X_tr, y_tr, log_transform=log_tf, verbose=False)
            prog.progress(55, text="5-fold CV 중...")
            cv = cross_validate(X_tr, y_tr, "gbr", 5, log_tf, verbose=False)
            prog.progress(80, text="성능 평가 중...")

            metrics = {}
            for mname, mt in models.items():
                metrics[mname] = {
                    "train": evaluate(mt, X_tr,  y_tr),
                    "val":   evaluate(mt, X_val, y_val),
                    "test":  evaluate(mt, X_te,  y_te),
                }
            prog.progress(95, text="AD 계산 중...")
            h_te, h_star = calc_leverage(X_tr, X_te)
            prog.progress(100, text="완료!")

        result = {
            "splits":  {"X_tr": X_tr, "X_val": X_val, "X_te": X_te,
                        "y_tr": y_tr, "y_val": y_val, "y_te": y_te},
            "models":  models,
            "cv":      cv,
            "metrics": metrics,
            "leverage":{"h_te": h_te, "h_star": h_star},
            "log_tf":  log_tf,
        }
        st.session_state["ml_result"] = result
        prog.empty()
        st.success("✅ 학습 완료!")
        _render_s3_results()


def _render_s3_results():
    res  = st.session_state["ml_result"]
    mets = res["metrics"]
    cv   = res["cv"]
    spl  = res["splits"]

    # ── 성능 지표 테이블 ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 모델 성능 비교")

    rows = []
    for mname in ["gbr","rf","ridge"]:
        for split in ["train","val","test"]:
            m = mets[mname][split]
            rows.append({
                "Model": mname.upper(),
                "Split": split,
                "RMSE":     m["RMSE"],
                "MAE":      m["MAE"],
                "R²":       m["R2"],
                "logRMSE":  m.get("log_RMSE", "-"),
            })
    perf_df = pd.DataFrame(rows)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # ── CV 요약 ───────────────────────────────────────────────────────────────
    st.markdown("### 5-Fold Cross-Validation  (GBR)")
    c1, c2, c3 = st.columns(3)
    c1.metric("CV RMSE",  f"{cv['rmse_mean']:.3f} ± {cv['rmse_std']:.3f}")
    c2.metric("CV MAE",   f"{cv['mae_mean']:.3f}")
    c3.metric("CV R²",    f"{cv['r2_mean']:.3f} ± {cv['r2_std']:.3f}")

    st.markdown("---")

    # ── 그래프 2열 배치 ───────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        # Actual vs Predicted (GBR, test set)
        st.markdown("**Actual vs Predicted  —  GBR (Test)**")
        y_true = mets["gbr"]["test"]["y_true"]
        y_pred = mets["gbr"]["test"]["y_pred"]

        fig, ax = plt.subplots(figsize=(5, 4))
        colors_sc = np.where(y_true < 2.4, "#27ae60",
                    np.where(y_true < 10.0, "#f39c12", "#e74c3c"))
        ax.scatter(y_true, y_pred, c=colors_sc, s=40, alpha=0.85, edgecolors="white", lw=0.5)
        lo = min(y_true.min(), y_pred.min()) * 0.9
        hi = max(y_true.max(), y_pred.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")
        ax.set_xlabel("Actual k")
        ax.set_ylabel("Predicted k")
        ax.set_title(f"GBR Test  R²={mets['gbr']['test']['R2']:.3f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        patches = [
            mpatches.Patch(color="#27ae60", label="k < 3"),
            mpatches.Patch(color="#f39c12", label="3 ≤ k < 10"),
            mpatches.Patch(color="#e74c3c", label="k ≥ 10"),
        ]
        ax.legend(handles=patches, fontsize=7)
        plt.tight_layout()
        _fig_to_st(fig, "양 축 log scale")

    with col_r:
        # CV fold 결과 bar chart
        st.markdown("**5-Fold CV R² per fold**")
        folds = range(1, 6)
        r2s   = cv["r2_folds"]
        colors_cv = ["#27ae60" if v >= 0.55 else "#f39c12" if v >= 0.40 else "#e74c3c"
                     for v in r2s]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(folds, r2s, color=colors_cv, edgecolor="white", alpha=0.9)
        ax.axhline(cv["r2_mean"], color="navy", ls="--", lw=1.5,
                   label=f"Mean R² = {cv['r2_mean']:.3f}")
        ax.set_xticks(list(folds))
        ax.set_xticklabels([f"Fold {i}" for i in folds])
        ax.set_ylabel("R²")
        ax.set_ylim(0, 1)
        ax.set_title("Cross-validation R² (GBR)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        _fig_to_st(fig)

    # ── 잔차 플롯 + Williams plot ─────────────────────────────────────────────
    st.markdown("---")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        # 잔차 플롯
        st.markdown("**Residual Plot  (Test)**")
        y_true_all = np.concatenate([
            mets["gbr"]["train"]["y_true"],
            mets["gbr"]["val"]["y_true"],
            mets["gbr"]["test"]["y_true"],
        ])
        y_pred_all = np.concatenate([
            mets["gbr"]["train"]["y_pred"],
            mets["gbr"]["val"]["y_pred"],
            mets["gbr"]["test"]["y_pred"],
        ])
        residuals = y_pred - y_true  # test only
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(y_true, residuals, c=colors_sc, s=30, alpha=0.8, edgecolors="none")
        ax.axhline(0, color="black", lw=1.2, ls="--")
        ax.set_xlabel("Actual k")
        ax.set_ylabel("Predicted - Actual")
        ax.set_title("Residuals (Test set)")
        ax.set_xscale("log")
        plt.tight_layout()
        _fig_to_st(fig)

    with col_r2:
        # Williams plot (AD)
        st.markdown("**Williams Plot  (Applicability Domain)**")
        h_te   = res["leverage"]["h_te"]
        h_star = res["leverage"]["h_star"]

        std_res = np.zeros_like(y_pred)
        rmse_te = mets["gbr"]["test"]["RMSE"]
        if rmse_te > 0:
            std_res = (y_pred - y_true) / rmse_te

        colors_ad = np.where(h_te <= h_star, "#3498db", "#e74c3c")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(h_te, std_res, c=colors_ad, s=35, alpha=0.85, edgecolors="white", lw=0.4)
        ax.axvline(h_star,  color="red",   ls="--", lw=1.5, label=f"h* = {h_star:.3f}")
        ax.axhline(3.0,     color="orange", ls="--", lw=1,  label="±3σ")
        ax.axhline(-3.0,    color="orange", ls="--", lw=1)
        ax.set_xlabel("Leverage h")
        ax.set_ylabel("Standardized Residual")
        ax.set_title("Williams Plot (AD)")
        n_in  = (h_te <= h_star).sum()
        n_out = (h_te >  h_star).sum()
        ax.legend(fontsize=8,
                  title=f"AD inside: {n_in}  outside: {n_out}")
        plt.tight_layout()
        _fig_to_st(fig, "파란점=AD 내부  빨간점=AD 외부(외삽 주의)")

    # ── 모델별 성능 비교 + 최적 모델 표시 ───────────────────────────────────────
    st.markdown("---")
    best = cv.get("best_model", "gbr")
    st.markdown("**📊 모델별 성능 비교 (Test R²)**")
    col_m = st.columns(4)
    for i, mname in enumerate(["gbr", "rf", "ridge", "gpr"]):
        if mname in mets:
            te = mets[mname]["test"]
            tr = mets[mname]["train"]
            label = f"{'⭐ ' if mname==best else ''}{mname.upper()}"
            col_m[i].metric(
                label,
                f"Test R²={te['R2']:.3f}",
                f"Train R²={tr['R2']:.3f}"
            )

    best_te = mets[best]["test"]
    best_tr = mets[best]["train"]
    st.success(
        f"**최적 모델: {best.upper()}**  |  "
        f"Train R²={best_tr['R2']:.3f}  "
        f"Test R²={best_te['R2']:.3f}  "
        f"Test logRMSE={best_te.get('log_RMSE','N/A')}  "
        f"CV R²={cv['r2_mean']:.3f}±{cv['r2_std']:.3f}"
    )

    st.session_state["s3_done"] = True


# ─────────────────────────────────────────────────────────────────────────────
# S4 — Candidate Screening
# ─────────────────────────────────────────────────────────────────────────────
def render_s4():
    st.subheader("🏆 Candidate Screening")

    if "ml_result" not in st.session_state:
        st.warning("⚠️ S3 탭에서 모델을 먼저 학습하세요.")
        return
    if "feat_names" not in st.session_state:
        st.warning("⚠️ S2 탭에서 피처를 먼저 계산하세요.")
        return

    models     = st.session_state["ml_result"]["models"]
    X_tr       = st.session_state["ml_result"]["splits"]["X_tr"]
    feat_names = st.session_state["feat_names"]

    # ── 입력 방식 선택 ────────────────────────────────────────────────────────
    st.markdown("**입력 방식 선택**")
    mode = st.radio("", ["📋 내장 후보 목록 사용", "✏️ 직접 SMILES 입력"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "📋 내장 후보 목록 사용":
        candidates = load_example_candidates()
        st.info(f"문헌 기반 low-k 후보 {len(candidates)}개가 로드됩니다.")
        default_text = "\n".join(f"{n},{s}" for n, s in candidates)
    else:
        default_text = (
            "# Name,SMILES 형식으로 입력 (한 줄씩)\n"
            "# 이름 없이 SMILES만 입력해도 됩니다\n"
            "Hexamethyldisiloxane,C[Si](C)(C)O[Si](C)(C)C\n"
            "Perfluorohexane,FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F\n"
        )

    smiles_text = st.text_area(
        "SMILES 입력 (Name,SMILES 또는 SMILES만)",
        value=default_text,
        height=200,
        key="s4_smiles_input",
    )

    # ── 스크리닝 실행 ─────────────────────────────────────────────────────────
    if st.button("🔍 스크리닝 실행", type="primary"):
        with st.spinner("기술자 계산 및 예측 중..."):
            df_result = screen(smiles_text, feat_names, models, X_tr)
        st.session_state["s4_result"] = df_result
        st.success(f"✅ 완료!  유효 분자: {df_result['valid'].sum()}  /  입력: {len(df_result)}")

    # ── 결과 표시 ─────────────────────────────────────────────────────────────
    if "s4_result" not in st.session_state:
        return

    df_result = st.session_state["s4_result"]
    df_valid  = df_result[df_result["valid"] == True].copy()

    if df_valid.empty:
        st.error("유효한 SMILES가 없습니다.")
        return

    st.markdown("---")

    # ── 요약 메트릭 ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("스크리닝 후보",     f"{len(df_valid)}개")
    c2.metric("k < 2.4 예측",     f"{(df_valid['pred_mean'] < 2.4).sum()}개")
    c3.metric("k < 2.0 예측",     f"{(df_valid['pred_mean'] < 2.0).sum()}개")
    c4.metric("AD 내부",          f"{df_valid['ad_ok'].sum()}개")

    st.markdown("---")

    # ── 탭: 결과 테이블 / 구조 그리드 / 비교 차트 ────────────────────────────
    t1, t2, t3 = st.tabs(["📋 결과 테이블", "🧪 구조 이미지", "📊 비교 차트"])

    with t1:
        st.markdown("**예측 결과 (pred_mean 오름차순)**")

        # 색상 포맷팅
        display_cols = ["rank","name","pred_mean","pred_std","pred_gbr",
                        "pred_rf","pred_ridge","h","ad_label"]
        df_disp = df_valid[display_cols].copy()
        df_disp.columns = ["순위","이름","예측k(평균)","불확실도",
                           "GBR","RF","Ridge","Leverage","AD"]

        def color_k(val):
            if val is None: return ""
            try:
                v = float(val)
                if v < 2.0: return "background-color: #d5f5e3"
                if v < 2.4: return "background-color: #fef9e7"
                if v < 5.0: return "background-color: #fdf2e9"
                return "background-color: #fadbd8"
            except: return ""

        styled = df_disp.style.map(color_k, subset=["예측k(평균)"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # CSV 다운로드
        csv = df_valid.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇ CSV 다운로드", csv,
                           "screening_results.csv", "text/csv")

    with t2:
        st.markdown("**Top-20 구조 (예측 k 오름차순)**")
        top20 = df_valid.head(20)
        grid_bytes = draw_grid(
            top20["smiles"].tolist(),
            top20["name"].tolist(),
            top20["pred_mean"].tolist(),
            n_cols=4,
        )
        if grid_bytes:
            st.image(grid_bytes, use_container_width=True)
        else:
            st.warning("구조 이미지 생성 실패")

        # 개별 구조 보기
        st.markdown("---")
        st.markdown("**개별 구조 확인**")
        sel_name = st.selectbox("분자 선택",
                                df_valid["name"].tolist(),
                                key="s4_mol_sel")
        sel_row = df_valid[df_valid["name"] == sel_name].iloc[0]
        svg_bytes = mol_to_png_bytes(sel_row["smiles"], size=300)
        if svg_bytes:
            col_img, col_info = st.columns([1, 2])
            with col_img:
                st.image(svg_bytes, width=280)
            with col_info:
                st.markdown(f"**{sel_name}**")
                st.markdown(f"- SMILES: `{sel_row['smiles']}`")
                st.markdown(f"- 예측 k (GBR): **{sel_row['pred_gbr']:.3f}**")
                st.markdown(f"- 앙상블 평균: **{sel_row['pred_mean']:.3f} ± {sel_row['pred_std']:.3f}**")
                st.markdown(f"- Leverage: {sel_row['h']:.4f}")
                st.markdown(f"- AD 상태: {sel_row['ad_label']}")

    with t3:
        st.markdown("**예측 k 분포  vs  훈련 데이터 Low-k 영역**")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # 왼쪽: 예측 k 분포
        ax = axes[0]
        ax.hist(df_valid["pred_mean"], bins=20,
                color="#3498db", edgecolor="white", alpha=0.85)
        ax.axvline(2.4, color="red",    ls="--", lw=1.5, label="k = 2.4")
        ax.axvline(2.0, color="orange", ls="--", lw=1.5, label="k = 2.0")
        ax.set_xlabel("Predicted k")
        ax.set_ylabel("Count")
        ax.set_title("Predicted k Distribution")
        ax.legend(fontsize=8)

        # 오른쪽: 산점도 (pred_mean vs pred_std, AD 색상)
        ax2 = axes[1]
        colors_ad = np.where(df_valid["ad_ok"], "#27ae60", "#e74c3c")
        sc = ax2.scatter(
            df_valid["pred_mean"],
            df_valid["pred_std"],
            c=colors_ad, s=60, alpha=0.85, edgecolors="white", lw=0.5
        )
        ax2.set_xlabel("Predicted k (mean)")
        ax2.set_ylabel("Uncertainty (std)")
        ax2.set_title("Prediction Uncertainty")
        patches = [
            mpatches.Patch(color="#27ae60", label="AD inside"),
            mpatches.Patch(color="#e74c3c", label="AD outside"),
        ]
        ax2.legend(handles=patches, fontsize=8)

        # 분자 이름 주석 (Top-5 low-k)
        top5 = df_valid.head(5)
        for _, row in top5.iterrows():
            ax2.annotate(row["name"][:12],
                         (row["pred_mean"], row["pred_std"]),
                         textcoords="offset points", xytext=(5, 3),
                         fontsize=6, alpha=0.8)

        plt.tight_layout()
        _fig_to_st(fig)

        # Top-10 요약 바 차트
        st.markdown("**Top-10 후보  (예측 k 최저값)**")
        top10 = df_valid.head(10)
        fig2, ax3 = plt.subplots(figsize=(9, 3.5))
        colors_bar = ["#27ae60" if v < 2.0 else "#f39c12" if v < 2.4 else "#3498db"
                      for v in top10["pred_mean"]]
        bars = ax3.barh(range(len(top10)), top10["pred_mean"].values[::-1],
                        color=colors_bar[::-1], edgecolor="white", alpha=0.9)
        ax3.errorbar(
            top10["pred_mean"].values[::-1],
            range(len(top10)),
            xerr=top10["pred_std"].values[::-1],
            fmt="none", color="gray", capsize=3, lw=1.2
        )
        ax3.set_yticks(range(len(top10)))
        ax3.set_yticklabels(
            [n[:20] for n in top10["name"].values[::-1]], fontsize=8
        )
        ax3.set_xlabel("Predicted k")
        ax3.axvline(2.4, color="red", ls="--", lw=1, label="k = 2.4")
        ax3.axvline(2.0, color="orange", ls="--", lw=1, label="k = 2.0")
        ax3.legend(fontsize=8)
        ax3.set_title("Top-10 Low-k Candidates")
        plt.tight_layout()
        _fig_to_st(fig2, "에러바 = 앙상블 불확실도 (3모델 표준편차)")

    st.session_state["s4_done"] = True


def render_s5():
    st.subheader("📄 Paper Draft")

    # ── 준비 상태 확인 ────────────────────────────────────────────────────────
    missing = []
    if "ml_result"   not in st.session_state: missing.append("S3 ML 학습")
    if "feat_names"  not in st.session_state: missing.append("S2 피처 계산")
    if "s4_result"   not in st.session_state: missing.append("S4 스크리닝")
    if missing:
        st.warning(f"⚠️ 먼저 완료하세요: {', '.join(missing)}")
        return

    df       = load_data()
    res      = st.session_state["ml_result"]
    feat_imp = st.session_state.get("feat_imp", None)

    if feat_imp is None:
        st.warning("⚠️ S2 탭을 다시 실행하여 피처 중요도를 계산하세요.")
        return

    # ── 논문 생성 버튼 ────────────────────────────────────────────────────────
    if "paper" not in st.session_state:
        if st.button("📝 논문 초안 생성", type="primary"):
            with st.spinner("실계산 수치 기반 논문 초안 생성 중..."):
                paper = generate_paper(
                    df_dataset   = df,
                    feat_names   = st.session_state["feat_names"],
                    X_all_shape  = st.session_state["X_all"].shape,
                    cv_result    = res["cv"],
                    metrics      = res["metrics"],
                    df_screening = st.session_state["s4_result"],
                    feat_imp     = feat_imp,
                    log_tf       = res.get("log_tf", True),
                )
                st.session_state["paper"] = paper

                # ── 자동 저장 ────────────────────────────────────────────────
                saved_paths = _autosave_paper(paper, st.session_state)
                st.session_state["paper_saved_paths"] = saved_paths

            st.success("✅ 논문 초안 생성 완료!")
        return

    paper = st.session_state["paper"]

    # ── 저장 경로 표시 ───────────────────────────────────────────────────────
    saved = st.session_state.get("paper_saved_paths", {})
    if saved:
        with st.expander("💾 자동 저장 위치", expanded=True):
            for label, path in saved.items():
                st.markdown(f"- **{label}**: `{path}`")

    # 재생성 버튼
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    if col_btn1.button("🔄 재생성"):
        st.session_state.pop("paper", None)
        st.rerun()

    # ── 탭 구성 ───────────────────────────────────────────────────────────────
    pt1, pt2, pt3, pt4, pt5 = st.tabs(
        ["📋 Abstract", "🔧 Methods", "📊 Results", "🏁 Conclusion", "📄 Full MD"]
    )

    with pt1:
        st.markdown(paper["abstract"])
    with pt2:
        st.markdown(paper["methods"])
    with pt3:
        st.markdown(paper["results"])
    with pt4:
        st.markdown(paper["conclusion"])
    with pt5:
        st.markdown("**전체 Markdown (복사 → Word/LaTeX 변환)**")
        st.text_area("", value=paper["full_md"], height=600, key="full_md_area")

    # ── 다운로드 ──────────────────────────────────────────────────────────────
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)

    col_d1.download_button(
        "⬇ Markdown 다운로드 (.md)",
        data     = paper["full_md"].encode("utf-8"),
        file_name= "lowk_qspr_paper.md",
        mime     = "text/markdown",
    )

    # 핵심 수치 요약 CSV
    gbr_te = res["metrics"]["gbr"]["test"]
    cv     = res["cv"]
    df_sc  = st.session_state["s4_result"]
    df_valid = df_sc[df_sc["valid"] == True] if "valid" in df_sc.columns else df_sc

    summary_data = {
        "항목":  ["데이터셋 분자수", "피처 수", "GBR Test R²", "GBR Test RMSE",
                   "GBR Test logRMSE", "CV R²", "스크리닝 후보수", "k<3 예측수"],
        "값":    [
            len(load_data()),
            len(st.session_state["feat_names"]),
            round(gbr_te["R2"],   4),
            round(gbr_te["RMSE"], 4),
            gbr_te.get("log_RMSE", "–"),
            f"{cv['r2_mean']:.3f}±{cv['r2_std']:.3f}",
            len(df_valid),
            int((df_valid["pred_mean"] < 2.4).sum()) if "pred_mean" in df_valid.columns else "–",
        ],
    }
    summary_csv = pd.DataFrame(summary_data).to_csv(index=False).encode("utf-8-sig")
    col_d2.download_button(
        "⬇ 핵심 수치 CSV",
        data      = summary_csv,
        file_name = "key_metrics.csv",
        mime      = "text/csv",
    )

    st.session_state["s5_done"] = True


# ─────────────────────────────────────────────────────────────────────────────
# 메인 레이아웃
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Low-k QSPR Platform",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # 헤더
    st.markdown(
        "<h2 style='margin-bottom:0'>⚡ Low-k Dielectric QSPR Platform</h2>"
        "<p style='color:gray;margin-top:4px'>CRC Handbook 234 molecules · RDKit · scikit-learn GBR</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # 진행 상황 표시
    s1_ok = "✅" if st.session_state.get("s1_done") else "○"
    s2_ok = "✅" if st.session_state.get("s2_done") else "○"
    s3_ok = "✅" if st.session_state.get("s3_done") else "○"
    s4_ok = "✅" if st.session_state.get("s4_done") else "○"
    s5_ok = "✅" if st.session_state.get("s5_done") else "○"
    st.caption(
        f"{s1_ok} S1 Dataset  →  {s2_ok} S2 Features  →  "
        f"{s3_ok} S3 ML Model  →  {s4_ok} S4 Screening  →  {s5_ok} S5 Paper"
    )

    # 탭
    tabs = st.tabs([
        "📊 S1. Dataset",
        "🔬 S2. Features",
        "🤖 S3. ML Model",
        "🏆 S4. Screening",
        "📄 S5. Paper",
    ])

    with tabs[0]: render_s1()
    with tabs[1]: render_s2()
    with tabs[2]: render_s3()
    with tabs[3]: render_s4()
    with tabs[4]: render_s5()


if __name__ == "__main__":
    main()
