# -*- coding: utf-8 -*-
"""
build_literature_dataset.py
────────────────────────────
논문 / 데이터시트 출처 Low-k 소재 데이터셋 구축

각 항목: (SMILES, k_exp, name, category, source_short, doi, note)
SMILES = 반복단위(repeat unit) 또는 대표 소분자 구조
k_exp  = 논문 보고 실험값 (25°C, 1 kHz 기준; 다른 조건은 note에 기재)

실행: python build_literature_dataset.py
결과: literature_dielectric.csv
      crc_dielectric_combined.csv  (CRC+PubChem+Literature 전체 병합)
"""

import os
import pandas as pd
from rdkit import Chem

# ─────────────────────────────────────────────────────────────────────────────
# 논문 기반 데이터
# (SMILES, k_exp, name, category, source_short, doi, note)
# ─────────────────────────────────────────────────────────────────────────────
LITERATURE_DATA = [

    # ── BCB (Benzocyclobutene / Cyclotene) ────────────────────────────────────
    ("C1CC2=CC=CC=C2C1",
     2.65, "BCB (Cyclotene, 1 MHz)", "bcb_parylene",
     "Microelectron. Eng. 1996",
     "10.1016/S0167931796000615",
     "DVS-BCB fully cured; 1 MHz, RT"),

    ("C1CC2=CC=CC=C2C1",
     2.49, "BCB (Cyclotene, 11-65 GHz)", "bcb_parylene",
     "J. Infrared Millim. Terahertz Waves 2009",
     "10.1007/s10762-009-9552-0",
     "CBCPW method, 11-65 GHz average"),

    # ── Parylene 계열 ─────────────────────────────────────────────────────────
    ("Cc1ccc(C)cc1",
     2.65, "Parylene N (1 kHz)", "bcb_parylene",
     "SCS Parylene datasheet",
     "",
     "CVD film; repeat unit = p-xylylene; 60Hz/1kHz/1MHz 모두 2.65"),

    ("Cc1ccc(C)c(Cl)c1",
     3.10, "Parylene C (1 kHz)", "bcb_parylene",
     "SCS Parylene datasheet",
     "",
     "CVD film; mono-chloro p-xylylene; 1 kHz"),

    ("Cc1cc(Cl)cc(C)c1Cl",
     2.82, "Parylene D (1 kHz)", "bcb_parylene",
     "SCS Parylene datasheet",
     "",
     "CVD film; di-chloro p-xylylene; 1 kHz"),

    ("FC(F)(C1CC=CC1)F",
     2.30, "Parylene AF4 (J. Fluorine Chem.)", "bcb_parylene",
     "J. Fluorine Chem. 2003",
     "10.1016/S0022-1139(03)00100-3",
     "All-fluorinated CVD parylene; in/out-of-plane avg"),

    ("FC(F)(C1CC=CC1)F",
     2.17, "Parylene HT (1 MHz)", "bcb_parylene",
     "IEEE ECTC 2023",
     "10.1109/ECTC51909.2023.00177",
     "Parylene HT = purified AF4; 1 MHz, Df=0.001"),

    ("FC(F)(C1CC=CC1)F",
     2.40, "Parylene VT4 (0.1Hz-100kHz)", "bcb_parylene",
     "Mater. Chem. Phys. 2013",
     "10.1016/j.matchemphys.2013.07.036",
     "Fluorinated variant; stable across freq/temp range"),

    # ── SiLK (Polyphenylene) ──────────────────────────────────────────────────
    ("c1ccc(-c2ccccc2)cc1",
     2.65, "SiLK polyphenylene (1 MHz)", "low_k_special",
     "MRS Proc. 1997",
     "10.1557/PROC-476-9",
     "Spin-on crosslinked polyphenylene; 1 MHz"),

    # ── Teflon AF (amorphous fluoropolymer) ───────────────────────────────────
    ("FC1(F)OC(F)(F)OC1(C(F)(F)F)C(F)(F)F",
     1.93, "Teflon AF 1600/2400", "fluorinated",
     "DuPont datasheet / ResearchGate",
     "",
     "Copolymer of PTFE + perfluorodioxole; stable to GHz"),

    ("FC1(F)OC(F)(F)OC1(C(F)(F)F)C(F)(F)F",
     1.57, "Porous Teflon AF (1 MHz)", "fluorinated",
     "Mater. Sci. Eng. B 2001",
     "10.1016/S0921-5107(01)00504-9",
     "Porous AF1600 film; 1 MHz"),

    # ── CYTOP (AGC amorphous perfluoropolymer) ────────────────────────────────
    ("FC1(F)C(F)(F)OC(F)(F)C(F)(F)C1(F)F",
     2.10, "CYTOP", "fluorinated",
     "AGC Chemicals datasheet",
     "",
     "Cyclic perfluoro ether polymer; volume resistivity >1e17 Ω·m"),

    # ── MSQ / PSSQ (methylsilsesquioxane) ────────────────────────────────────
    ("C[Si](O)(O)O",
     2.70, "MSQ dense film", "silicon",
     "Mol. Cryst. Liq. Cryst. 2001",
     "10.1080/10587250108024702",
     "Dense methylsilsesquioxane spin-on; k~2.7"),

    ("C[Si](O)(O)O",
     2.20, "Porous MSQ (30 vol%)", "silicon",
     "Mol. Cryst. Liq. Cryst. 2001",
     "10.1080/10587250108024702",
     "30 vol% porosity introduced; k~2.2"),

    ("C[Si](O)(O)O",
     2.10, "MSQ + D4 porogen", "silicon",
     "Vacuum 2013",
     "10.1016/j.vacuum.2013.01.030",
     "D4 as sacrificial porogen; k~2.1"),

    # ── HSQ (hydrogen silsesquioxane) ─────────────────────────────────────────
    ("[SiH](O)(O)O",
     2.85, "HSQ dense (cured 400°C)", "silicon",
     "Thin Solid Films 1998",
     "10.1016/S0040-6090(98)01041-0",
     "Spin-on HSQ; MOS capacitor; cured at 400°C/1h"),

    ("[SiH](O)(O)O",
     2.20, "Porous HSQ (D4 porogen)", "silicon",
     "J. Am. Ceram. Soc. 2007",
     "10.1111/j.1551-2916.2007.01713.x",
     "D4 sacrificial pores; k~2.2; 1kHz-1GHz stable"),

    # ── POSS 계열 ─────────────────────────────────────────────────────────────
    # BCB-POSS: POSS 케이지 대신 phenyl-silsesquioxane 단위 (대표 구조)
    ("c1ccc([Si](O)(O)O)cc1",
     2.24, "BCB-POSS T8 (phenyl-POSS unit)", "silicon",
     "Polymer 2023",
     "10.1016/j.polymer.2023.126189",
     "8-cage BCB-POSS; phenylsilanetriol as repeat unit proxy; 1 kHz"),

    ("c1ccc([Si](O)(O)O)cc1",
     2.02, "BCB-POSS T10 (phenyl-POSS unit)", "silicon",
     "Polymer 2023",
     "10.1016/j.polymer.2023.126189",
     "10-cage BCB-POSS; phenylsilanetriol as repeat unit proxy; 1 kHz"),

    ("FC(F)(F)[Si](O)(O)O",
     2.47, "PI + fluorinated POSS (2 wt%)", "silicon",
     "Polym. Eng. Sci. 2022",
     "10.1002/pen.26063",
     "Trifluoropropyl POSS blended in polyimide; 1 MHz"),

    # ── 6FDA 계열 불소 폴리이미드 ──────────────────────────────────────────────
    # 6FDA 다이안하이드라이드 (대표 구조: 플루오린화 프탈산 무수물)
    ("O=C1OC(=O)c2ccc(C(c3ccc4c(c3)C(=O)OC4=O)(C(F)(F)F)C(F)(F)F)cc21",
     2.72, "6FDA-TFMB polyimide (10 GHz)", "fluorinated",
     "ACS Appl. Polym. Mater. 2021",
     "10.1021/acsapm.0c01141",
     "6FDA dianhydride unit; TFMB diamine; 10 GHz, Df=0.0075"),

    ("O=C1OC(=O)c2ccc(C(c3ccc4c(c3)C(=O)OC4=O)(C(F)(F)F)C(F)(F)F)cc21",
     1.84, "6FDA-TFMB ultra-low-k PI", "fluorinated",
     "Eur. Polym. J. 2021",
     "10.1016/j.eurpolymj.2020.109904",
     "Micro-branched crosslink + fluorination; 10^4 Hz"),

    # ── 불소 폴리노르보르넨 ───────────────────────────────────────────────────
    ("FC(F)(OCC1CC2CC1CC2)F",
     2.39, "F-polynorbornene (5 GHz)", "fluorinated",
     "ACS Appl. Polym. Mater. 2020",
     "10.1021/acsapm.9b01070",
     "Cross-linked via -OCF=CF2 groups; 5 GHz, Df=3.7e-3"),

    # ── CDO / SiOC:H (porous organosilicate) ─────────────────────────────────
    ("C[Si](O)(C)O",
     3.00, "Dense SiOC:H (CDO)", "silicon",
     "J. Appl. Phys. 2008",
     "10.1063/1.2937082",
     "PECVD SiOC:H dense film; CH3 termination"),

    ("C[Si](O)(C)O",
     2.30, "Porous SiOCH (32nm node)", "silicon",
     "Academia.edu / ITRS",
     "",
     "30% porosity; 32 nm technology node target"),

    ("C[Si](O)(C)O",
     2.50, "PECVD porous organosilicate (UV)", "silicon",
     "Surf. Coat. Technol. 2007",
     "10.1016/j.surfcoat.2007.05.036",
     "UV post-treatment; 24-31% porosity"),

    ("C[Si](O)(C)O",
     1.50, "Nanoporous PMSSQ (dendrimer)", "silicon",
     "Nature Materials 2005",
     "10.1038/nmat1291",
     "Dendritic sphere imprinting; pore <2nm; k~1.5"),

    # ── 주요 리뷰 논문 대표 수치 (Maier 2001) ─────────────────────────────────
    ("Nc1ccc(-c2ccc(N)cc2)cc1",
     3.40, "Aromatic polyimide (ODA-PMDA)", "polymer",
     "Maier, Prog. Polym. Sci. 2001",
     "10.1016/S0079-6700(00)00043-5",
     "Representative unfluorinated PI; benchmark"),

    ("FC(F)(F)c1ccc(N)cc1",
     2.80, "Fluorinated PI (CF3 pendant)", "fluorinated",
     "Maier, Prog. Polym. Sci. 2001",
     "10.1016/S0079-6700(00)00043-5",
     "CF3 substituted aromatic PI; lower k vs. unfluorinated"),
]


# ─────────────────────────────────────────────────────────────────────────────
def build_dataset() -> pd.DataFrame:
    rows = []
    valid, invalid = 0, 0

    for entry in LITERATURE_DATA:
        smi, k, name, cat, src, doi, note = entry
        mol = Chem.MolFromSmiles(smi)
        ok  = mol is not None
        if ok:
            valid += 1
        else:
            invalid += 1
            print(f"  [INVALID SMILES] {name}: {smi}")

        rows.append({
            "smiles":      smi,
            "k_exp":       k,
            "name":        name,
            "category":    cat,
            "source":      src,
            "doi":         doi,
            "note":        note,
            "rdkit_valid": ok,
            "smiles_src":  "literature",
        })

    df = pd.DataFrame(rows)
    print(f"\nLiterature dataset: {valid} valid / {invalid} invalid / {len(df)} total")
    return df[df["rdkit_valid"]].copy()


def merge_all(base_csv: str, lit_df: pd.DataFrame) -> pd.DataFrame:
    """CRC+PubChem 데이터와 Literature 데이터 병합"""
    base = pd.read_csv(base_csv)

    # doi, note 컬럼 없으면 추가
    for col in ["doi", "note"]:
        if col not in base.columns:
            base[col] = ""

    # 이름 기준 중복 제거 (base 우선)
    combined = pd.concat([base, lit_df], ignore_index=True)
    before   = len(combined)
    combined = combined.drop_duplicates(subset="name", keep="first")
    after    = len(combined)
    combined = combined.sort_values("k_exp").reset_index(drop=True)

    print(f"Merge: base {len(base)} + literature {len(lit_df)} "
          f"→ {before} → dedup {after}  (removed {before-after})")
    return combined


def print_stats(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  Combined Dataset Stats (CRC + PubChem + Literature)")
    print("="*60)
    print(f"  Total     : {len(df)}")
    print(f"  k range   : {df['k_exp'].min():.2f} ~ {df['k_exp'].max():.2f}")
    print(f"  k mean    : {df['k_exp'].mean():.2f}")
    print()
    src_counts = df["source"].value_counts()
    print("  Source breakdown:")
    for src, cnt in src_counts.head(10).items():
        bar = "#" * (cnt // 2)
        print(f"    {src[:35]:<35} {cnt:>3}  {bar}")
    print()
    print(f"  k < 3.0 (Low-k)      : {(df['k_exp']<3.0).sum()}")
    print(f"  k < 2.5 (Ultra low-k): {(df['k_exp']<2.5).sum()}")
    print(f"  k < 2.0 (Super low-k): {(df['k_exp']<2.0).sum()}")
    print("="*60)


if __name__ == "__main__":
    base_dir   = os.path.dirname(__file__)
    base_csv   = os.path.join(base_dir, "crc_dielectric_full.csv")
    lit_csv    = os.path.join(base_dir, "literature_dielectric.csv")
    comb_csv   = os.path.join(base_dir, "crc_dielectric_combined.csv")

    assert os.path.exists(base_csv), f"Base CSV not found: {base_csv}"

    # 1. 논문 데이터셋 구축
    lit_df = build_dataset()
    lit_df.to_csv(lit_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {lit_csv}")

    # 2. 전체 병합
    combined = merge_all(base_csv, lit_df)
    print_stats(combined)

    combined.to_csv(comb_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {comb_csv}  ({len(combined)} molecules)")
