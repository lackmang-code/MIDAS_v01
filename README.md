# MIDAS v01
### Materials Informatics for Design and Automated Screening

> **단일 물성(유전율) 기반 저유전율 유기막 재료 탐색 및 논문 초안 자동 생성 플랫폼**

---

## Overview

MIDAS v01 is a QSPR (Quantitative Structure-Property Relationship) platform for  
predicting and screening **low-k dielectric organic thin film materials**.  
All results are based on real computation — no LLM-generated hallucinations.

```
SMILES → RDKit Descriptors → ML Model (GBR) → Screening → Paper Draft
```

---

## Features

| Tab | Function |
|-----|----------|
| 📊 S1. Dataset | EDA · 263 molecules · literature sources · DOI table · add custom data |
| 🔬 S2. Features | RDKit 233 descriptors → 50 selected (variance + corr + RF importance) |
| 🤖 S3. ML Model | GBR / RF / Ridge · 5-fold CV · Williams plot (AD) · Actual vs Predicted |
| 🏆 S4. Screening | Ensemble prediction · uncertainty · applicability domain · structure images |
| 📄 S5. Paper | Auto-generated draft with real computed numbers · Markdown export |

---

## Dataset

| Source | Count | Notes |
|--------|-------|-------|
| CRC Handbook "Permittivity of Liquids" | 215 | 25°C experimental values |
| Polymer Handbook | 19 | Supplemental values |
| Literature (papers + datasheets) | 29 | BCB, Parylene, SiLK, POSS, 6FDA-PI, etc. |
| **Total** | **263** | k range: 1.50 ~ 108.94 |

**k < 3.0 (Low-k candidates): 138 molecules**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run app (Windows)
run.bat

# Or directly
streamlit run app.py --server.port 8502
```

Access: `http://localhost:8502`

---

## Requirements

```
streamlit >= 1.32
pandas >= 2.0
numpy >= 1.24
rdkit >= 2023.3
scikit-learn >= 1.3
matplotlib >= 3.7
```

---

## Workflow

```
Step 1: S1 탭 — 데이터셋 확인 및 논문 출처 검토
Step 2: S2 탭 — RDKit 분자 기술자 계산 (클릭 1번, ~5초)
Step 3: S3 탭 — ML 모델 학습 (클릭 1번, ~15초)
Step 4: S4 탭 — 후보 분자 스크리닝 (내장 30개 or 직접 입력)
Step 5: S5 탭 — 논문 초안 생성 → output/ 폴더 자동 저장
```

---

## Model Performance (example)

| Model | Test R² | Test RMSE | CV R² |
|-------|---------|-----------|-------|
| GBR (primary) | ~0.71 | ~11.5 | ~0.54 ± 0.10 |
| RandomForest | ~0.54 | ~14.7 | — |
| Ridge | ~0.67 | ~12.4 | — |

*263 molecules, 50 features, stratified 70/15/15 split*

---

## Key Papers (Literature Sources)

- Maier, G. *Prog. Polym. Sci.* **2001**, 26, 3–65. `10.1016/S0079-6700(00)00043-5`
- Volksen et al. *Chem. Rev.* **2010**, 110, 56–110. `10.1021/cr9002819`
- BCB/Cyclotene: *Microelectron. Eng.* **1996** `10.1016/S0167931796000615`
- Parylene AF4: *J. Fluorine Chem.* **2003** `10.1016/S0022-1139(03)00100-3`
- Nanoporous MSQ: *Nature Mater.* **2005** `10.1038/nmat1291`

---

## Project Structure

```
MIDAS_v01/
├── app.py                    # Streamlit main app
├── run.bat                   # Windows launcher
├── requirements.txt
├── data/
│   ├── crc_dielectric_combined.csv   # Main dataset (263 mols)
│   ├── build_crc_dataset.py          # CRC data builder
│   ├── fetch_pubchem_supplement.py   # PubChem supplement
│   └── build_literature_dataset.py   # Literature data builder
├── engines/
│   ├── feature_engine.py     # RDKit descriptor calculation
│   ├── ml_engine.py          # Training / CV / prediction
│   ├── screening_engine.py   # Candidate screening + AD
│   └── paper_engine.py       # Paper draft generation
└── output/                   # Auto-saved results
```

---

## Roadmap

- **MIDAS v01** ← *here* — Single property (dielectric constant)
- **MIDAS v02** — Multi-property platform (k + modulus + viscosity)

---

## License

MIT License

---

*Built with RDKit · scikit-learn · Streamlit*
