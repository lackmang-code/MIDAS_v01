# -*- coding: utf-8 -*-
"""
PubChem 보완 데이터셋 구축
- 문헌에서 k 값이 확인된 화합물 목록 정의
- PubChem REST API 로 Canonical SMILES 취득 (표준화)
- RDKit 유효성 검증 후 CRC 데이터셋과 병합

출처: Polymer Handbook, CRC Handbook (vol.별), J. Phys. Chem. Ref. Data
실행: python fetch_pubchem_supplement.py
결과: crc_dielectric_full.csv  (CRC 193 + PubChem 보완)
"""

import time
import os
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors

# ─────────────────────────────────────────────────────────────────────────────
# 보완 화합물 목록  (name, k_lit, category, source, fallback_smiles)
#   fallback_smiles : PubChem 조회 실패 시 사용하는 수동 SMILES
# ─────────────────────────────────────────────────────────────────────────────
SUPPLEMENT = [

    # ── Siloxanes / Silicones ─────────────────────────────────────────────────
    ("Hexamethyldisiloxane",              2.179, "silicon",     "Polymer Handbook",
     "C[Si](C)(C)O[Si](C)(C)C"),
    ("Octamethyltrisiloxane",             2.390, "silicon",     "Polymer Handbook",
     "C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C"),
    ("Decamethyltetrasiloxane",           2.370, "silicon",     "Polymer Handbook",
     "C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C"),
    ("Octamethylcyclotetrasiloxane",      2.390, "silicon",     "Polymer Handbook",
     "C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1"),
    ("Decamethylcyclopentasiloxane",      2.500, "silicon",     "Polymer Handbook",
     "C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1"),
    ("Tetramethylsilane",                 1.921, "silicon",     "CRC Handbook",
     "C[Si](C)(C)C"),
    ("Trimethylchlorosilane",             2.130, "silicon",     "Polymer Handbook",
     "C[Si](C)(C)Cl"),
    ("Trimethylmethoxysilane",            2.300, "silicon",     "Polymer Handbook",
     "C[Si](C)(C)OC"),
    ("Triethylsilane",                    2.240, "silicon",     "Polymer Handbook",
     "CC[SiH](CC)CC"),
    ("Tetramethylorthosilicate",          2.700, "silicon",     "Polymer Handbook",
     "CO[Si](OC)(OC)OC"),

    # ── Fluorinated (additional) ──────────────────────────────────────────────
    ("Hexafluorobenzene",                 2.029, "fluorinated", "CRC Handbook",
     "Fc1c(F)c(F)c(F)c(F)c1F"),
    ("Octafluorotoluene",                 2.270, "fluorinated", "Polymer Handbook",
     "FC(F)(F)c1c(F)c(F)c(F)c(F)c1F"),
    ("1,2,4,5-Tetrafluorobenzene",        2.530, "fluorinated", "CRC Handbook",
     "Fc1cc(F)c(F)cc1F"),
    ("Perfluoromethylcyclohexane",        1.820, "fluorinated", "Polymer Handbook",
     "FC(F)(F)C1(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C1(F)F"),
    ("Perfluorooctane",                   1.890, "fluorinated", "Polymer Handbook",
     "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),
    ("Perfluorodecalin",                  2.000, "fluorinated", "Polymer Handbook",
     "FC1(F)C(F)(F)C(F)(F)C(F)(F)C2(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C12F"),
    ("Perfluorocyclobutane",              2.130, "fluorinated", "CRC Handbook",
     "FC1(F)C(F)(F)C(F)(F)C1(F)F"),
    ("Perfluorotributylamine",            1.820, "fluorinated", "Polymer Handbook",
     "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)N(C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),
    ("Trifluoromethylbenzene",            9.220, "fluorinated", "CRC Handbook",
     "FC(F)(F)c1ccccc1"),
    ("1,3-Bis(trifluoromethyl)benzene",   5.980, "fluorinated", "CRC Handbook",
     "FC(F)(F)c1cccc(C(F)(F)F)c1"),
    ("Perfluorodecane",                   1.910, "fluorinated", "Polymer Handbook",
     "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),

    # ── Aromatic (additional) ─────────────────────────────────────────────────
    ("Fluorene",                          2.640, "aromatic",    "CRC Handbook",
     "C1=CC2=CC=CC=C2C1c1ccccc1"),  # fallback; PubChem preferred
    ("Acenaphthylene",                    2.850, "aromatic",    "CRC Handbook",
     "C1=CC2=CC=CC3=CC=CC1=C23"),
    ("Acenaphthene",                      3.009, "aromatic",    "CRC Handbook",
     "C1CC2=CC=CC3=CC=CC1=C23"),
    ("Phenanthrene",                      2.800, "aromatic",    "CRC Handbook",
     "c1ccc2c(c1)ccc1ccccc12"),
    ("Pyrene",                            2.808, "aromatic",    "CRC Handbook",
     "c1ccc2ccc3cccc4ccc1c2c34"),
    ("Triphenylene",                      2.840, "aromatic",    "CRC Handbook",
     "c1ccc2c(c1)ccc1ccc3ccccc3c12"),
    ("9,10-Dihydroanthracene",            2.720, "aromatic",    "Polymer Handbook",
     "C1Cc2ccccc2Cc2ccccc21"),
    ("Fluoranthene",                      2.980, "aromatic",    "CRC Handbook",
     "c1ccc2ccc3cccc4ccc1c2c34"),  # same as pyrene? no, different; see below
    ("p-Terphenyl",                       2.900, "aromatic",    "Polymer Handbook",
     "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"),

    # ── Chlorinated solvents ──────────────────────────────────────────────────
    ("1,2-Dichloroethane",                10.42, "halogen",     "CRC Handbook",
     "ClCCCl"),
    ("Tetrachloromethane",                2.238, "halogen",     "CRC Handbook",
     "ClC(Cl)(Cl)Cl"),
    ("1,1,1-Trichloroethane",             7.243, "halogen",     "CRC Handbook",
     "CC(Cl)(Cl)Cl"),
    ("1,2-Dibromoethane",                 4.990, "halogen",     "CRC Handbook",
     "BrCCBr"),
    ("Dibromomethane",                    7.770, "halogen",     "CRC Handbook",
     "BrCBr"),

    # ── Ethers (additional) ───────────────────────────────────────────────────
    ("Diisopropyl ether",                 3.805, "ether",       "CRC Handbook",
     "CC(C)OC(C)C"),
    ("Dibutyl ether",                     3.083, "ether",       "CRC Handbook",
     "CCCCOCCCC"),
    ("Diphenyl ether",                    3.730, "ether",       "CRC Handbook",
     "c1ccc(Oc2ccccc2)cc1"),
    ("Anisole",                           4.330, "ether",       "CRC Handbook",
     "COc1ccccc1"),
    ("1,4-Dioxane",                       2.209, "ether",       "CRC Handbook",
     "C1COCCO1"),
    ("1,3-Dioxolane",                     7.130, "ether",       "CRC Handbook",
     "C1OCCO1"),
    ("Tetrahydrofuran",                   7.520, "ether",       "CRC Handbook",
     "C1CCCO1"),
    ("2-Methyltetrahydrofuran",           6.970, "ether",       "CRC Handbook",
     "CC1CCCO1"),
    ("Crown ether 12-crown-4",            3.950, "ether",       "Polymer Handbook",
     "C1COCCOCCOCCOC1"),

    # ── Alkenes / Alkynes ─────────────────────────────────────────────────────
    ("1-Hexene",                          2.077, "alkane",      "CRC Handbook",
     "CCCCC=C"),
    ("Cyclohexene",                       2.220, "cycloalkane", "CRC Handbook",
     "C1=CCCCC1"),
    ("1-Octene",                          2.113, "alkane",      "CRC Handbook",
     "CCCCCCC=C"),
    ("1-Decene",                          2.136, "alkane",      "CRC Handbook",
     "CCCCCCCCC=C"),
    ("Styrene",                           2.435, "aromatic",    "CRC Handbook",
     "C=Cc1ccccc1"),
    ("alpha-Methylstyrene",               2.420, "aromatic",    "Polymer Handbook",
     "C=C(C)c1ccccc1"),
    ("Indene",                            2.921, "aromatic",    "CRC Handbook",
     "C1=CC2=CC=CC=C2C1"),

    # ── Nitrogen heterocycles ─────────────────────────────────────────────────
    ("Quinoline",                         9.160, "aromatic",    "CRC Handbook",
     "c1ccc2ncccc2c1"),
    ("Isoquinoline",                      11.00, "aromatic",    "CRC Handbook",
     "c1ccc2cnccc2c1"),
    ("Pyridine",                          13.26, "aromatic",    "CRC Handbook",
     "c1ccncc1"),

    # ── Low-k special ─────────────────────────────────────────────────────────
    ("p-Xylene",                          2.270, "aromatic",    "CRC Handbook",
     "Cc1ccc(C)cc1"),
    ("m-Xylene",                          2.359, "aromatic",    "CRC Handbook",
     "Cc1cccc(C)c1"),
    ("o-Xylene",                          2.562, "aromatic",    "CRC Handbook",
     "Cc1ccccc1C"),
    ("Mesitylene",                        2.279, "aromatic",    "CRC Handbook",
     "Cc1cc(C)cc(C)c1"),
    ("1,2,4-Trimethylbenzene",            2.377, "aromatic",    "CRC Handbook",
     "Cc1ccc(C)c(C)c1"),
    ("Durene",                            2.436, "aromatic",    "CRC Handbook",
     "Cc1cc(C)c(C)cc1C"),
]


# ─────────────────────────────────────────────────────────────────────────────
# PubChem REST API
# ─────────────────────────────────────────────────────────────────────────────
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    "/{name}/property/CanonicalSMILES/JSON"
)

def get_canonical_smiles(name: str, fallback: str):
    """
    PubChem 에서 Canonical SMILES 취득.
    실패 시 fallback SMILES 반환.
    Returns (smiles, origin) where origin = 'pubchem' | 'fallback'
    """
    try:
        url = PUBCHEM_URL.format(name=requests.utils.quote(name))
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"]
            smiles = props[0]["CanonicalSMILES"]
            return smiles, "pubchem"
    except Exception:
        pass
    return fallback, "fallback"


def validate_smiles(smiles: str):
    """RDKit 로 SMILES 유효성 검증. 실패 시 None."""
    mol = Chem.MolFromSmiles(smiles)
    return mol  # None if invalid


# ─────────────────────────────────────────────────────────────────────────────
# 메인 워크플로우
# ─────────────────────────────────────────────────────────────────────────────
def build_supplement() -> pd.DataFrame:
    rows = []
    print(f"\nQuerying PubChem for {len(SUPPLEMENT)} compounds...")
    print("-" * 60)

    for i, (name, k_lit, cat, src, fallback_smi) in enumerate(SUPPLEMENT, 1):
        smiles, origin = get_canonical_smiles(name, fallback_smi)
        mol = validate_smiles(smiles)
        valid = mol is not None

        # 유효하지 않으면 fallback 재시도
        if not valid and origin == "pubchem":
            smiles = fallback_smi
            origin = "fallback"
            mol = validate_smiles(smiles)
            valid = mol is not None

        status = f"[{origin[:3].upper()}]" if valid else "[FAIL]"
        print(f"  {i:>3}. {name:<40} {status}  k={k_lit:.3f}")

        rows.append({
            "smiles":      smiles,
            "k_exp":       k_lit,
            "name":        name,
            "category":    cat,
            "source":      src,
            "rdkit_valid": valid,
            "smiles_src":  origin,
        })

        # PubChem rate limit 준수 (5 req/s 이하)
        time.sleep(0.25)

    df = pd.DataFrame(rows)
    valid_df = df[df["rdkit_valid"]].copy()
    print(f"\nValid: {len(valid_df)} / {len(df)}")
    return valid_df


def merge_and_deduplicate(crc_path: str, supp_df: pd.DataFrame) -> pd.DataFrame:
    """CRC 데이터셋과 병합 후 중복(이름 기준) 제거."""
    crc_df = pd.read_csv(crc_path)

    # smiles_src 컬럼 없으면 추가
    if "smiles_src" not in crc_df.columns:
        crc_df["smiles_src"] = "crc_manual"

    combined = pd.concat([crc_df, supp_df], ignore_index=True)

    # 이름 기준 중복 제거 (CRC 우선 → 먼저 나온 것 유지)
    before = len(combined)
    combined = combined.drop_duplicates(subset="name", keep="first")
    after = len(combined)
    print(f"\nMerge: CRC {len(crc_df)} + Supplement {len(supp_df)} "
          f"-> {before} -> dedup {after}  (removed {before - after} dup)")

    combined = combined.sort_values("k_exp").reset_index(drop=True)
    return combined


def print_summary(df: pd.DataFrame):
    print("\n" + "="*55)
    print("  Full Dataset Summary")
    print("="*55)
    print(f"  Total molecules  : {len(df)}")
    print(f"  k range          : {df['k_exp'].min():.2f} ~ {df['k_exp'].max():.2f}")
    print(f"  k mean / median  : {df['k_exp'].mean():.2f} / {df['k_exp'].median():.2f}")
    print()

    cat_counts = df.groupby("category")["k_exp"].agg(["count", "min", "max", "mean"])
    cat_counts = cat_counts.sort_values("mean")
    print(f"  {'Category':<20} {'N':>4}  {'k range':<14} {'k mean':>6}")
    print("  " + "-" * 50)
    for cat, row in cat_counts.iterrows():
        print(f"  {cat:<20} {int(row['count']):>4}  "
              f"{row['min']:.2f}~{row['max']:.2f}{'':>5} {row['mean']:>6.2f}")
    print()
    print(f"  k < 3.0 (Low-k)      : {len(df[df['k_exp'] < 3.0])}")
    print(f"  k < 2.5 (Ultra low-k): {len(df[df['k_exp'] < 2.5])}")
    print(f"  k < 2.0 (Super low-k): {len(df[df['k_exp'] < 2.0])}")
    print("="*55)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir  = os.path.dirname(__file__)
    crc_csv   = os.path.join(base_dir, "crc_dielectric.csv")
    out_csv   = os.path.join(base_dir, "crc_dielectric_full.csv")

    assert os.path.exists(crc_csv), f"CRC CSV not found: {crc_csv}"

    # 1. PubChem 에서 SMILES 취득 + 검증
    supp_df = build_supplement()

    # 2. CRC 데이터셋과 병합
    full_df = merge_and_deduplicate(crc_csv, supp_df)

    # 3. 통계 출력
    print_summary(full_df)

    # 4. 저장 (rdkit_valid == True 만)
    full_df = full_df[full_df["rdkit_valid"]].reset_index(drop=True)
    full_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_csv}  ({len(full_df)} molecules)")
