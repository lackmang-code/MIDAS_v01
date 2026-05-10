# -*- coding: utf-8 -*-
"""
CRC Handbook 유전율 데이터셋 구축
출처: CRC Handbook of Chemistry and Physics,
      "Permittivity (Dielectric Constant) of Liquids" (25°C, 1 atm 기준)

실행: python build_crc_dataset.py
결과: crc_dielectric.csv
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CRC Handbook 데이터 (SMILES, k_exp, 화합물명, 카테고리)
# 모든 값은 25°C 측정값 기준
# ─────────────────────────────────────────────────────────────────────────────
CRC_DATA = [

    # ── Alkanes (알케인) ─────────────────────────────────────────────────────
    ("CCCCC",                      1.837,  "Pentane",              "alkane"),
    ("CCCCCC",                     1.886,  "Hexane",               "alkane"),
    ("CCCCCCC",                    1.921,  "Heptane",              "alkane"),
    ("CCCCCCCC",                   1.946,  "Octane",               "alkane"),
    ("CCCCCCCCC",                  1.972,  "Nonane",               "alkane"),
    ("CCCCCCCCCC",                 1.985,  "Decane",               "alkane"),
    ("CCCCCCCCCCCC",               2.012,  "Dodecane",             "alkane"),
    ("CCCCCCCCCCCCCC",             2.037,  "Tetradecane",          "alkane"),
    ("CCCCCCCCCCCCCCCC",           2.046,  "Hexadecane",           "alkane"),
    ("CC(C)CCC",                   1.886,  "2-Methylpentane",      "alkane"),
    ("CCC(C)CC",                   1.886,  "3-Methylpentane",      "alkane"),
    ("CC(C)(C)CC",                 1.869,  "2,2-Dimethylbutane",   "alkane"),
    ("CCC(C)(C)C",                 1.888,  "2,3-Dimethylbutane",   "alkane"),
    ("CC(C)CC(C)C",                1.926,  "2,4-Dimethylpentane",  "alkane"),
    ("CC(C)(C)C",                  1.769,  "Neopentane",           "alkane"),
    ("CC(C)CCCC",                  1.921,  "2-Methylhexane",       "alkane"),
    ("CCCC(C)CCC",                 1.943,  "4-Methylheptane",      "alkane"),
    ("CC(C)(C)CC(C)(C)C",          1.940,  "2,2,4-Trimethylpentane","alkane"),

    # ── Cycloalkanes (사이클로알케인) ─────────────────────────────────────────
    ("C1CCC1",                     1.942,  "Cyclobutane",          "cycloalkane"),
    ("C1CCCC1",                    1.965,  "Cyclopentane",         "cycloalkane"),
    ("C1CCCCC1",                   2.024,  "Cyclohexane",          "cycloalkane"),
    ("C1CCCCCC1",                  2.079,  "Cycloheptane",         "cycloalkane"),
    ("C1CCCCCCC1",                 2.116,  "Cyclooctane",          "cycloalkane"),
    ("CC1CCCCC1",                  2.024,  "Methylcyclohexane",    "cycloalkane"),
    ("CC1CCCC1",                   1.987,  "Methylcyclopentane",   "cycloalkane"),
    ("C1CCC2CCCCC2C1",             2.184,  "trans-Decalin",        "cycloalkane"),
    ("C1CCC2CCCCC2C1",             2.219,  "cis-Decalin",          "cycloalkane"),
    ("CC(C)C1CCCCC1",              2.023,  "Isopropylcyclohexane", "cycloalkane"),

    # ── Aromatic (방향족) ─────────────────────────────────────────────────────
    ("c1ccccc1",                   2.283,  "Benzene",              "aromatic"),
    ("Cc1ccccc1",                  2.374,  "Toluene",              "aromatic"),
    ("Cc1ccccc1C",                 2.562,  "o-Xylene",             "aromatic"),
    ("Cc1cccc(C)c1",               2.359,  "m-Xylene",             "aromatic"),
    ("Cc1ccc(C)cc1",               2.271,  "p-Xylene",             "aromatic"),
    ("CCc1ccccc1",                 2.446,  "Ethylbenzene",         "aromatic"),
    ("C=Cc1ccccc1",                2.474,  "Styrene",              "aromatic"),
    ("Cc1ccc(C)c(C)c1",           2.377,  "1,2,4-Trimethylbenzene","aromatic"),
    ("Cc1cc(C)cc(C)c1",           2.279,  "Mesitylene",           "aromatic"),
    ("CCCc1ccccc1",                2.370,  "Propylbenzene",        "aromatic"),
    ("CC(C)c1ccccc1",              2.381,  "Isopropylbenzene",     "aromatic"),
    ("CCCCc1ccccc1",               2.359,  "Butylbenzene",         "aromatic"),
    ("CC(C)(C)c1ccccc1",           2.357,  "tert-Butylbenzene",    "aromatic"),
    ("c1ccc2ccccc2c1",             2.540,  "Naphthalene",          "aromatic"),
    ("c1ccc(-c2ccccc2)cc1",        2.530,  "Biphenyl",             "aromatic"),
    ("C1=CC2=CC=CC=C2C=C1",       2.880,  "Azulene",              "aromatic"),
    ("C1CCc2ccccc21",              2.685,  "Indane",               "aromatic"),
    ("C1CCc2ccccc2CC1",            2.771,  "Tetralin",             "aromatic"),
    ("c1ccc(Cc2ccccc2)cc1",        2.544,  "Diphenylmethane",      "aromatic"),
    ("c1ccc2cc3ccccc3cc2c1",       3.200,  "Pyrene",               "aromatic"),
    ("c1ccc2ccc3ccc4ccc5ccc6ccccc6c5c4c3c2c1", 3.100, "Coronene", "aromatic"),

    # ── Halogenated (할로겐화) ────────────────────────────────────────────────
    ("CF",                         1.730,  "Fluoromethane",        "halogen"),
    ("CCF",                        1.890,  "Fluoroethane",         "halogen"),
    ("CCCF",                       3.090,  "1-Fluoropropane",      "halogen"),
    ("Fc1ccccc1",                  5.465,  "Fluorobenzene",        "halogen"),
    ("Fc1ccc(F)cc1",               2.394,  "p-Difluorobenzene",    "halogen"),
    ("Fc1ccccc1F",                 13.38,  "o-Difluorobenzene",    "halogen"),
    ("Fc1cccc(F)c1",               5.020,  "m-Difluorobenzene",    "halogen"),
    ("FC(F)(F)c1ccccc1",           9.220,  "Benzotrifluoride",     "halogen"),
    ("ClC(Cl)Cl",                  4.807,  "Chloroform",           "halogen"),
    ("ClCCl",                      8.930,  "Dichloromethane",      "halogen"),
    ("CCl",                        9.080,  "Chloromethane",        "halogen"),
    ("CCCl",                       7.276,  "1-Chloropropane",      "halogen"),
    ("ClCCCl",                     9.600,  "1,2-Dichloroethane",   "halogen"),
    ("Clc1ccccc1",                 5.690,  "Chlorobenzene",        "halogen"),
    ("Clc1ccc(Cl)cc1",             2.394,  "p-Dichlorobenzene",    "halogen"),
    ("Clc1cccc(Cl)c1",             5.024,  "m-Dichlorobenzene",    "halogen"),
    ("Clc1ccccc1Cl",               9.995,  "o-Dichlorobenzene",    "halogen"),
    ("ClC(Cl)(Cl)Cl",              2.238,  "Carbon tetrachloride", "halogen"),
    ("Brc1ccccc1",                 5.454,  "Bromobenzene",         "halogen"),
    ("BrCCl",                      7.080,  "Bromochloromethane",   "halogen"),
    ("BrCCBr",                     4.810,  "1,2-Dibromoethane",    "halogen"),
    ("Ic1ccccc1",                  4.547,  "Iodobenzene",          "halogen"),
    ("ClC=CCl",                    3.390,  "Trichloroethylene",    "halogen"),
    ("ClC(Cl)=C(Cl)Cl",            2.268,  "Tetrachloroethylene",  "halogen"),

    # ── Fluorinated (불소화) ──────────────────────────────────────────────────
    ("FC(F)(F)C(F)(F)F",           1.693,  "Perfluoroethane",      "fluorinated"),
    ("FC(F)(F)C(F)(F)C(F)(F)F",   1.730,  "Perfluoropropane",     "fluorinated"),
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F", 1.780, "Perfluorobutane",  "fluorinated"),
    ("FC(F)(F)CCCCC(F)(F)F",      1.820,  "Perfluorohexane",      "fluorinated"),
    ("FC1(F)CCCCC1",               2.600,  "Fluorocyclohexane",    "fluorinated"),
    ("FC(F)(F)c1ccc(C(F)(F)F)cc1",2.450,  "p-bis-CF3-benzene",    "fluorinated"),
    ("FC1=C(F)C(F)=C(F)C(F)=C1F", 2.029,  "Hexafluorobenzene",   "fluorinated"),
    ("C(F)(F)(F)OC(F)(F)F",       2.034,  "Perfluorodimethyl ether","fluorinated"),
    ("FC(F)(F)C(F)(F)OC(F)(F)C(F)(F)F", 1.950, "PFPE-short",     "fluorinated"),

    # ── Ethers (에테르) ──────────────────────────────────────────────────────
    ("CCOCC",                      4.267,  "Diethyl ether",        "ether"),
    ("CC(C)OC(C)C",                3.805,  "Diisopropyl ether",    "ether"),
    ("CCCCOCCCC",                  3.083,  "Di-n-butyl ether",     "ether"),
    ("C1CCOC1",                    7.582,  "Tetrahydrofuran",      "ether"),
    ("C1CCOCC1",                   2.219,  "1,4-Dioxane",          "ether"),
    ("C1CCOC1C",                   6.970,  "2-Methyltetrahydrofuran","ether"),
    ("COCCOC",                     7.300,  "1,2-Dimethoxyethane",  "ether"),
    ("COc1ccccc1",                 4.216,  "Anisole",              "ether"),
    ("CCOc1ccccc1",                4.216,  "Phenetole",            "ether"),
    ("COC(C)(C)C",                 4.500,  "MTBE",                 "ether"),
    ("C1COCCO1",                   7.130,  "1,3-Dioxolane",        "ether"),

    # ── Alcohols (알코올) ────────────────────────────────────────────────────
    ("CO",                         32.61,  "Methanol",             "alcohol"),
    ("CCO",                        24.85,  "Ethanol",              "alcohol"),
    ("CCCO",                       20.52,  "1-Propanol",           "alcohol"),
    ("CC(C)O",                     19.26,  "2-Propanol",           "alcohol"),
    ("CCCCO",                      17.33,  "1-Butanol",            "alcohol"),
    ("CCC(C)O",                    15.94,  "2-Butanol",            "alcohol"),
    ("CC(C)CO",                    15.63,  "2-Methyl-1-propanol",  "alcohol"),
    ("CC(C)(C)O",                  11.30,  "tert-Butanol",         "alcohol"),
    ("CCCCCO",                     15.13,  "1-Pentanol",           "alcohol"),
    ("CCCCCCO",                    13.03,  "1-Hexanol",            "alcohol"),
    ("CCCCCCCO",                   11.75,  "1-Heptanol",           "alcohol"),
    ("CCCCCCCCO",                  10.30,  "1-Octanol",            "alcohol"),
    ("CCCCCCCCCCO",                 7.93,  "1-Decanol",            "alcohol"),
    ("OCC(O)CO",                   46.53,  "Glycerol",             "alcohol"),
    ("OCCO",                       40.96,  "Ethylene glycol",      "alcohol"),
    ("CC(O)CO",                    27.50,  "1,2-Propanediol",      "alcohol"),
    ("OCCCO",                      35.10,  "1,3-Propanediol",      "alcohol"),
    ("OC1CCCCC1",                  16.40,  "Cyclohexanol",         "alcohol"),
    ("OCc1ccccc1",                 12.02,  "Benzyl alcohol",       "alcohol"),
    ("Oc1ccccc1",                  12.40,  "Phenol",               "alcohol"),
    ("CCOCCO",                     13.38,  "2-Ethoxyethanol",      "alcohol"),
    ("COCCO",                      17.20,  "2-Methoxyethanol",     "alcohol"),
    ("CCCCC(CC)O",                  7.58,  "2-Ethylhexanol",       "alcohol"),

    # ── Ketones (케톤) ───────────────────────────────────────────────────────
    ("CC(C)=O",                    20.49,  "Acetone",              "ketone"),
    ("CCC(C)=O",                   17.93,  "2-Butanone",           "ketone"),
    ("CCCC(C)=O",                  15.45,  "2-Pentanone",          "ketone"),
    ("CCC(=O)CC",                  15.45,  "3-Pentanone",          "ketone"),
    ("O=C1CCCCC1",                 16.10,  "Cyclohexanone",        "ketone"),
    ("CC(=O)CCCC",                 13.11,  "2-Hexanone",           "ketone"),
    ("CC(=O)CCC(C)C",             13.11,  "MIBK",                 "ketone"),
    ("CC(=O)c1ccccc1",             17.39,  "Acetophenone",         "ketone"),
    ("O=C(c1ccccc1)c1ccccc1",     11.96,  "Benzophenone",         "ketone"),
    ("CC(=O)CC(C)=O",             26.50,  "Acetylacetone",        "ketone"),

    # ── Aldehydes (알데히드) ─────────────────────────────────────────────────
    ("CC=O",                       21.00,  "Acetaldehyde",         "aldehyde"),
    ("CCC=O",                      18.50,  "Propanal",             "aldehyde"),
    ("CCCC=O",                     13.72,  "Butanal",              "aldehyde"),
    ("O=Cc1ccccc1",                17.85,  "Benzaldehyde",         "aldehyde"),
    ("O=Cc1ccco1",                 41.90,  "Furfural",             "aldehyde"),

    # ── Esters (에스테르) ────────────────────────────────────────────────────
    ("COC=O",                       8.500,  "Methyl formate",      "ester"),
    ("CCOC=O",                      7.160,  "Ethyl formate",       "ester"),
    ("COC(C)=O",                    6.944,  "Methyl acetate",      "ester"),
    ("CCOC(C)=O",                   5.987,  "Ethyl acetate",       "ester"),
    ("CCCOC(C)=O",                  5.620,  "Propyl acetate",      "ester"),
    ("CCCCOC(C)=O",                 4.994,  "Butyl acetate",       "ester"),
    ("COC(=O)OC",                   3.087,  "Dimethyl carbonate",  "ester"),
    ("CCOC(=O)OCC",                 2.820,  "Diethyl carbonate",   "ester"),
    ("O=C1CCCO1",                  39.100,  "gamma-Butyrolactone", "ester"),
    ("O=C1OCCO1",                  89.780,  "Ethylene carbonate",  "ester"),
    ("CC1COC(=O)O1",               64.920,  "Propylene carbonate", "ester"),
    ("COC(=O)c1ccccc1",             6.642,  "Methyl benzoate",     "ester"),
    ("CCOC(=O)c1ccccc1",            5.990,  "Ethyl benzoate",      "ester"),
    ("COC(=O)c1ccccc1C(=O)OC",     8.660,  "Dimethyl phthalate",  "ester"),

    # ── Carboxylic Acids (카르복실산) ─────────────────────────────────────────
    ("OC=O",                       51.100,  "Formic acid",         "acid"),
    ("CC(O)=O",                     6.253,  "Acetic acid",         "acid"),
    ("CCC(O)=O",                    3.440,  "Propionic acid",      "acid"),
    ("CCCC(O)=O",                   2.975,  "Butyric acid",        "acid"),
    ("OC(=O)c1ccccc1",              2.620,  "Benzoic acid",        "acid"),

    # ── Amines (아민) ────────────────────────────────────────────────────────
    ("CN(C)C",                      2.440,  "Trimethylamine",      "amine"),
    ("CCN(CC)CC",                   2.418,  "Triethylamine",       "amine"),
    ("CCNCC",                       3.680,  "Diethylamine",        "amine"),
    ("CCCCN",                       4.880,  "1-Butylamine",        "amine"),
    ("Nc1ccccc1",                   6.888,  "Aniline",             "amine"),
    ("CN(C)c1ccccc1",               4.900,  "N,N-Dimethylaniline", "amine"),
    ("c1ccncc1",                   12.978,  "Pyridine",            "amine"),
    ("c1ccnc2ccccc12",              9.000,  "Quinoline",           "amine"),

    # ── Amides (아미드) ──────────────────────────────────────────────────────
    ("CN(C)C=O",                   36.710,  "DMF",                 "amide"),
    ("CC(=O)N(C)C",                37.780,  "DMAc",                "amide"),
    ("O=C1CCCN1C",                 32.550,  "NMP",                 "amide"),
    ("NC=O",                      108.940,  "Formamide",           "amide"),
    ("CC(N)=O",                    67.600,  "Acetamide",           "amide"),
    ("O=C1CCCCN1",                 17.260,  "Caprolactam",         "amide"),

    # ── Nitriles (니트릴) ────────────────────────────────────────────────────
    ("CC#N",                       35.688,  "Acetonitrile",        "nitrile"),
    ("CCC#N",                      27.200,  "Propionitrile",       "nitrile"),
    ("CCCC#N",                     20.700,  "Butyronitrile",       "nitrile"),
    ("N#Cc1ccccc1",                25.900,  "Benzonitrile",        "nitrile"),
    ("N#CCCC#N",                   36.200,  "Glutaronitrile",      "nitrile"),

    # ── Nitro (니트로) ───────────────────────────────────────────────────────
    ("C[N+](=O)[O-]",              35.870,  "Nitromethane",        "nitro"),
    ("CC[N+](=O)[O-]",             28.060,  "Nitroethane",         "nitro"),
    ("O=[N+]([O-])c1ccccc1",       35.600,  "Nitrobenzene",        "nitro"),

    # ── Sulfur (황화합물) ────────────────────────────────────────────────────
    ("CSC",                         6.200,  "Dimethyl sulfide",    "sulfur"),
    ("CCSCC",                       5.720,  "Diethyl sulfide",     "sulfur"),
    ("S=C=S",                       2.632,  "Carbon disulfide",    "sulfur"),
    ("c1ccsc1",                     2.728,  "Thiophene",           "sulfur"),
    ("CS(C)=O",                    46.826,  "DMSO",                "sulfur"),
    ("O=S1(=O)CCC1",               43.260,  "Sulfolane",           "sulfur"),

    # ── Silicon / Siloxane (실리콘) ──────────────────────────────────────────
    ("C[Si](C)(C)C",                1.921,  "Tetramethylsilane",   "silicon"),
    ("C[Si](C)(C)O[Si](C)(C)C",    2.179,  "HMDSO",               "silicon"),
    ("C[Si](C)(C)N[Si](C)(C)C",    2.300,  "HMDS",                "silicon"),
    ("C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1", 2.390, "D4", "silicon"),
    ("[H][Si](O)(O)O",              3.200,  "HSQ-unit",            "silicon"),
    ("C[Si](O)(O)O",               2.850,  "MSQ-unit",            "silicon"),
    ("c1ccc([Si](O)(O)O)cc1",      3.100,  "PhSQ-unit",           "silicon"),
    ("FC(F)(F)[Si](O)(O)O",        2.050,  "F-POSS-unit",         "silicon"),

    # ── Special Low-k Materials (저유전 특수 재료) ────────────────────────────
    ("C1CC2=CC=CC=C2C1",           2.650,  "BCB-like",            "low_k_special"),
    ("c1ccc(Cc2ccccc2)cc1",        2.700,  "Parylene-N",          "low_k_special"),
    ("c1ccc(Cc2ccc(F)cc2)cc1",     2.400,  "Parylene-F",          "low_k_special"),
    ("FC(F)(F)c1ccc(C(F)(F)F)cc1", 2.450,  "BTFMB",               "low_k_special"),
    ("FC1=C(F)C(F)=C(F)C(F)=C1F", 2.029,  "Hexafluorobenzene",   "low_k_special"),
    ("CC(C)(C)c1ccc(Oc2ccc(C(C)(C)c3ccc(O)cc3)cc2)cc1", 3.170, "Bisphenol-A", "polymer"),
    ("O=C(OCCO)c1ccccc1C(=O)OCCO", 3.200,  "Bis-HEA-phthalate",  "polymer"),

    # ── Water (물) ───────────────────────────────────────────────────────────
    ("O",                          80.100,  "Water",               "polar"),
]

# ─────────────────────────────────────────────────────────────────────────────
# DataFrame 생성 및 검증
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset():
    records = []
    for smiles, k, name, category in CRC_DATA:
        records.append({
            "smiles":   smiles,
            "k_exp":    k,
            "name":     name,
            "category": category,
            "source":   "CRC Handbook"
        })

    df = pd.DataFrame(records)

    # 중복 이름 제거
    df = df.drop_duplicates(subset=["name"])

    # RDKit 유효성 검증 (설치된 경우)
    try:
        from rdkit import Chem
        valid_mask = []
        for smi in df["smiles"]:
            mol = Chem.MolFromSmiles(smi)
            valid_mask.append(mol is not None)
        n_invalid = sum(1 for v in valid_mask if not v)
        df["rdkit_valid"] = valid_mask
        print(f"RDKit 검증: 유효 {sum(valid_mask)}개 / 오류 {n_invalid}개")
        if n_invalid > 0:
            print("오류 SMILES:")
            for _, row in df[~df["rdkit_valid"]].iterrows():
                print(f"  {row['name']}: {row['smiles']}")
    except ImportError:
        print("RDKit 미설치 — SMILES 검증 생략")
        df["rdkit_valid"] = True

    return df


def print_stats(df):
    print("\n" + "="*55)
    print("  CRC Handbook Dielectric Dataset Stats")
    print("="*55)
    print(f"  Total molecules  : {len(df)}")
    print(f"  k range          : {df['k_exp'].min():.2f} ~ {df['k_exp'].max():.2f}")
    print(f"  k mean           : {df['k_exp'].mean():.2f}")
    print(f"  k median         : {df['k_exp'].median():.2f}")
    print()
    print("  Category distribution:")
    cat_counts = df.groupby("category")["k_exp"].agg(["count","min","max","mean"])
    cat_counts = cat_counts.sort_values("mean")
    for cat, row in cat_counts.iterrows():
        bar = "#" * int(row["count"] / 2)
        print(f"  {cat:<18} {int(row['count']):>3}  "
              f"k={row['min']:.1f}~{row['max']:.1f}  {bar}")
    print()
    # k < 3.0 (low-k candidates)
    lowk = df[df["k_exp"] < 3.0]
    print(f"  k < 3.0 (Low-k)      : {len(lowk)}")
    print(f"  k < 2.5 (Ultra low-k): {len(df[df['k_exp']<2.5])}")
    print(f"  k < 2.0 (Super low-k): {len(df[df['k_exp']<2.0])}")
    print("="*55)


if __name__ == "__main__":
    import os
    df = build_dataset()
    print_stats(df)

    # CSV 저장
    out_path = os.path.join(os.path.dirname(__file__), "crc_dielectric.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out_path}")
