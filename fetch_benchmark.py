"""Download and prepare the thirteen benchmark datasets under data/benchmark/.

The prepared files are bundled with the repository, so this script only
needs running to rebuild them. It fetches the raw files from the public
GitHub mirrors listed in SOURCES (the UCI and OpenML servers are not needed),
verifies their SHA-256 against the values recorded at preparation time,
applies the preparation steps documented in data/benchmark/PROVENANCE.md and
writes one CSV per dataset with the target as the last column, plus a
<key>-metadata.json with one description per feature, which is the only
dataset information the LLM prior ever sees.

Usage:  python fetch_benchmark.py            # download if missing, then prepare
        python fetch_benchmark.py --prepare  # prepare from data/benchmark/raw/ only
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "benchmark"
RAW = OUT / "raw"

# raw file -> (url, sha256). Files marked None are derived locally.
SOURCES = {
    "wine_quality/winequality-red.csv": (
        "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-red.csv",
        "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"),
    "wine_quality/winequality-white.csv": (
        "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-white.csv",
        "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836"),
    "dermatology/dermatology.arff": (
        "https://raw.githubusercontent.com/renatopp/arff-datasets/master/classification/dermatology.arff",
        "0d1c4236e524b90cc570c7c545c2c01d4043a3bba9ca0824b838e919771025ad"),
    "ionosphere/ionosphere.csv": (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv",
        "fd6dd7864b55d56dac0a1e6e24af9ccc35bf2555ac79af8ab9f3d1daa065ab83"),
    "spambase/spambase.data": (
        "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/nlp_class/spambase.data",
        "b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12"),
    "spambase/spambase.arff": (
        "https://raw.githubusercontent.com/renatopp/arff-datasets/master/classification/spambase.arff",
        "de045774b02909770aa52717398b1960258d2567669c35092e1d76f3c279b937"),
    "optdigits/optdigits.tsv.gz": (
        "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/optdigits/optdigits.tsv.gz",
        "b70b7ac68a54d1ccd41bd24f11324c3af96b67541791fb9d8684cbdf151533c3"),
    "geo_music/default_features_1059_tracks.txt": (
        "https://raw.githubusercontent.com/johannesharmse/Music_Origin_Prediction/master/data/default_features_1059_tracks.txt",
        "8557da3791e17b036763a95a7f16e3f23d20dcd4c5387d4f11aa1ed8c7a0b9ab"),
    "mice_protein/mice.csv": (
        "https://raw.githubusercontent.com/pycaret/datasets/main/data/common/mice.csv",
        "5ec490ae48bf7f2eb958fd33ef5036e58a2933cc05e816064c0ea3b14ad3903b"),
    "parkinsons/parkinsons.csv": (
        "https://raw.githubusercontent.com/pycaret/datasets/main/data/common/parkinsons.csv",
        "8dd8c084e51bd21fe360f910c10fc5d590113141c4803355c11591325f22df12"),
    "student_performance/student-por.csv": (
        "https://media.githubusercontent.com/media/WillKoehrsen/Data-Analysis/master/bayesian_lr/data/student-por.csv",
        "a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f"),
    "student_dropout/data.csv": (
        "https://raw.githubusercontent.com/Minhbui100/Predict-Students-Dropout/main/data.csv",
        "3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae"),
    "sonar/sonar.csv": (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv",
        "3079c09b5d2789a0f96aff82c28e5164fafe2495c5f8da96c6c256c1bd25763f"),
    "ames_housing/housing.csv": (
        "https://raw.githubusercontent.com/wblakecannon/ames/master/data/housing.csv",
        "1cf821e5ab53ce9ac177cb663872c2d2b8c5b385f85eed495ce62a213bd34f86"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download():
    for rel, (url, digest) in SOURCES.items():
        path = RAW / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"fetch {url}")
            urllib.request.urlretrieve(url, path)
        got = sha256(path)
        if not got.startswith(digest):
            raise SystemExit(f"{rel}: checksum {got[:16]} differs from the recorded {digest[:16]}")


def median_impute(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def write(key: str, df: pd.DataFrame, descriptions: dict):
    missing = [c for c in df.columns[:-1] if c not in descriptions]
    if missing:
        raise ValueError(f"{key}: no description for {missing}")
    df.to_csv(OUT / f"{key}.csv", index=False)
    (OUT / f"{key}-metadata.json").write_text(
        json.dumps({c: descriptions[c] for c in df.columns[:-1]}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"  {key}: {df.shape[0]} x {df.shape[1] - 1} + target")


# ------------------------------------------------------------------ metadata

WINE = {
    "fixed acidity": "tartaric acid (g/dm3), the non-volatile acids",
    "volatile acidity": "acetic acid (g/dm3); high levels give a vinegar taste",
    "citric acid": "citric acid (g/dm3), adds freshness",
    "residual sugar": "sugar left after fermentation (g/dm3)",
    "chlorides": "salt content (g/dm3)",
    "free sulfur dioxide": "free SO2 (mg/dm3), prevents microbial growth and oxidation",
    "total sulfur dioxide": "free and bound SO2 (mg/dm3)",
    "density": "density (g/cm3), depends on alcohol and sugar",
    "pH": "acidity on the pH scale",
    "sulphates": "potassium sulphate (g/dm3), an antimicrobial additive",
    "alcohol": "alcohol content (% by volume)",
    "colour": "red or white wine",
}

WDBC = {}
for stat, what in (("mean", "mean"), ("error", "standard error"), ("worst", "largest (mean of the three largest values)")):
    for feat, desc in (("radius", "distance from centre to perimeter"),
                       ("texture", "standard deviation of grey-scale values"),
                       ("perimeter", "nucleus perimeter"), ("area", "nucleus area"),
                       ("smoothness", "local variation in radius lengths"),
                       ("compactness", "perimeter^2 / area - 1"),
                       ("concavity", "severity of concave portions of the contour"),
                       ("concave points", "number of concave portions of the contour"),
                       ("symmetry", "symmetry of the nucleus"),
                       ("fractal dimension", "coastline approximation - 1")):
        name = f"{stat} {feat}" if stat != "error" else f"{feat} error"
        WDBC[name] = f"{what} of {desc} over the cell nuclei in the image"

DERM_CLINICAL = ["erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon",
                 "polygonal_papules", "follicular_papules", "oral_mucosal_involvement",
                 "knee_and_elbow_involvement", "scalp_involvement", "family_history"]
DERM_HISTO = ["melanin_incontinence", "eosinophils_in_the_infiltrate", "PNL_infiltrate",
              "fibrosis_of_the_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis",
              "parakeratosis", "clubbing_of_the_rete_ridges", "elongation_of_the_rete_ridges",
              "thinning_of_the_suprapapillary_epidermis", "spongiform_pustule", "munro_microabcess",
              "focal_hypergranulosis", "disappearance_of_the_granular_layer",
              "vacuolisation_and_damage_of_basal_layer", "spongiosis", "saw_tooth_appearance_of_retes",
              "follicular_horn_plug", "perifollicular_parakeratosis",
              "inflammatory_monoluclear_inflitrate", "band_like_infiltrate"]
DERM = {c: f"clinical attribute, graded 0-3: {c.replace('_', ' ')}" for c in DERM_CLINICAL}
DERM["family_history"] = "clinical attribute, 0 or 1: family history of the disease"
DERM.update({c: f"histopathological attribute from the skin biopsy, graded 0-3: {c.replace('_', ' ')}"
             for c in DERM_HISTO})
DERM["age"] = "age of the patient in years"

IONO = {f"attribute_{i:02d}": f"radar return attribute {i}: {'real' if i % 2 else 'imaginary'} part of "
        f"the complex autocorrelation of pulse number {(i + 1) // 2}" for i in range(1, 35)}

SPAM_WORDS = ["make", "address", "all", "3d", "our", "over", "remove", "internet", "order", "mail",
              "receive", "will", "people", "report", "addresses", "free", "business", "email", "you",
              "credit", "your", "font", "000", "money", "hp", "hpl", "george", "650", "lab", "labs",
              "telnet", "857", "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
              "meeting", "original", "project", "re", "edu", "table", "conference"]
SPAM = {f"word_freq_{w}": f"percentage of words in the e-mail that are '{w}'" for w in SPAM_WORDS}
SPAM.update({f"char_freq_{c}": f"percentage of characters in the e-mail that are '{c}'"
             for c in [";", "(", "[", "!", "$", "#"]})
SPAM.update({"capital_run_length_average": "average length of uninterrupted sequences of capital letters",
             "capital_run_length_longest": "length of the longest uninterrupted sequence of capital letters",
             "capital_run_length_total": "total number of capital letters in the e-mail"})

OPTD = {f"pixel_{r}_{c}": f"ink count (0-16) in cell row {r}, column {c} of the 8 x 8 bitmap "
        f"of a handwritten digit" for r in range(8) for c in range(8)}

GEO = {f"feature_{i:02d}": f"audio descriptor {i} of the recording (chromatic and timbral "
       f"features extracted with MARSYAS; no name is published)" for i in range(1, 69)}

MICE_NAMES = ['DYRK1A_N', 'ITSN1_N', 'BDNF_N', 'NR1_N', 'NR2A_N', 'pAKT_N', 'pBRAF_N', 'pCAMKII_N',
              'pCREB_N', 'pELK_N', 'pERK_N', 'pJNK_N', 'PKCA_N', 'pMEK_N', 'pNR1_N', 'pNR2A_N',
              'pNR2B_N', 'pPKCAB_N', 'pRSK_N', 'AKT_N', 'BRAF_N', 'CAMKII_N', 'CREB_N', 'ELK_N',
              'ERK_N', 'GSK3B_N', 'JNK_N', 'MEK_N', 'TRKA_N', 'RSK_N', 'APP_N', 'Bcatenin_N',
              'SOD1_N', 'MTOR_N', 'P38_N', 'pMTOR_N', 'DSCR1_N', 'AMPKA_N', 'NR2B_N', 'pNUMB_N',
              'RAPTOR_N', 'TIAM1_N', 'pP70S6_N', 'NUMB_N', 'P70S6_N', 'pGSK3B_N', 'pPKCG_N',
              'CDK5_N', 'S6_N', 'ADARB1_N', 'AcetylH3K9_N', 'RRP1_N', 'BAX_N', 'ARC_N', 'ERBB4_N',
              'nNOS_N', 'Tau_N', 'GFAP_N', 'GluR3_N', 'GluR4_N', 'IL1B_N', 'P3525_N', 'pCASP9_N',
              'PSD95_N', 'SNCA_N', 'Ubiquitin_N', 'pGSK3B_Tyr216_N', 'SHH_N', 'BAD_N', 'BCL2_N',
              'pS6_N', 'pCFOS_N', 'SYP_N', 'H3AcK18_N', 'EGR1_N', 'H3MeK4_N', 'CaNA_N']
MICE = {}
for n in MICE_NAMES:
    base = n[:-2]
    phospho = base.startswith("p") and base[1:2].isupper()
    MICE[n] = (f"expression level of {'phosphorylated ' if phospho else ''}{base[1:] if phospho else base}"
               f" in the nuclear fraction of mouse cortex (normalised)")
MICE["DYRK1A_N"] = ("expression level of DYRK1A, a chromosome-21 (trisomic) kinase, in the nuclear "
                    "fraction of mouse cortex")
MICE["ITSN1_N"] = "expression level of ITSN1 (intersectin 1), a chromosome-21 protein"
MICE["APP_N"] = "expression level of APP (amyloid precursor protein), a chromosome-21 protein"
MICE["SOD1_N"] = "expression level of SOD1 (superoxide dismutase 1), a chromosome-21 protein"
MICE["DSCR1_N"] = "expression level of DSCR1 (Down syndrome critical region 1), a chromosome-21 protein"
MICE["BDNF_N"] = "expression level of BDNF (brain-derived neurotrophic factor), a learning-related protein"
MICE["pCREB_N"] = "expression level of phosphorylated CREB, a memory-consolidation transcription factor"
MICE["pERK_N"] = "expression level of phosphorylated ERK (MAPK signalling, learning related)"
MICE["pCAMKII_N"] = "expression level of phosphorylated CaMKII (synaptic plasticity)"
MICE["ARC_N"] = "expression level of ARC (activity-regulated cytoskeleton protein, learning induced)"
MICE["EGR1_N"] = "expression level of EGR1 (immediate-early gene product, learning induced)"
MICE["pCFOS_N"] = "expression level of phosphorylated c-Fos (immediate-early gene product)"
MICE["GFAP_N"] = "expression level of GFAP (astrocyte marker)"
MICE["SYP_N"] = "expression level of synaptophysin (synaptic vesicle marker)"
MICE["PSD95_N"] = "expression level of PSD95 (postsynaptic density protein)"
MICE["Tau_N"] = "expression level of Tau"
MICE["H3AcK18_N"] = "acetylated histone H3 at lysine 18"
MICE["H3MeK4_N"] = "methylated histone H3 at lysine 4"
MICE["AcetylH3K9_N"] = "acetylated histone H3 at lysine 9"
MICE["CaNA_N"] = "expression level of calcineurin A"
MICE["pGSK3B_Tyr216_N"] = "expression level of GSK3B phosphorylated at Tyr216"
MICE["Ubiquitin_N"] = "ubiquitin level"
MICE["Bcatenin_N"] = "expression level of beta-catenin"
MICE["P3525_N"] = "expression level of P3525"
MICE["nNOS_N"] = "expression level of neuronal nitric oxide synthase"
MICE["IL1B_N"] = "expression level of interleukin 1 beta"
MICE["SHH_N"] = "expression level of sonic hedgehog"

PARK = {
    "age": "subject age in years", "sex": "subject sex (0 male, 1 female)",
    "test_time": "days since recruitment into the trial",
    "Jitter(%)": "frequency variation of the voice: jitter in percent",
    "Jitter(Abs)": "absolute jitter in microseconds",
    "Jitter:RAP": "relative average perturbation of pitch periods",
    "Jitter:PPQ5": "five-point period perturbation quotient",
    "Jitter:DDP": "average absolute difference of differences between consecutive periods",
    "Shimmer": "amplitude variation of the voice: local shimmer",
    "Shimmer(dB)": "shimmer in decibels",
    "Shimmer:APQ3": "three-point amplitude perturbation quotient",
    "Shimmer:APQ5": "five-point amplitude perturbation quotient",
    "Shimmer:APQ11": "eleven-point amplitude perturbation quotient",
    "Shimmer:DDA": "average absolute difference between consecutive amplitude differences",
    "NHR": "noise-to-harmonics ratio", "HNR": "harmonics-to-noise ratio",
    "RPDE": "recurrence period density entropy, a nonlinear dynamical complexity measure",
    "DFA": "detrended fluctuation analysis, signal fractal scaling exponent",
    "PPE": "pitch period entropy, a nonlinear measure of fundamental frequency variation",
}

STUDENT = {
    "school": "student's school (GP Gabriel Pereira, MS Mousinho da Silveira)",
    "sex": "student's sex", "age": "student's age (15-22)",
    "address": "home address type (U urban, R rural)",
    "famsize": "family size (LE3 three or fewer, GT3 more than three)",
    "Pstatus": "parents' cohabitation status (T together, A apart)",
    "Medu": "mother's education (0 none to 4 higher education)",
    "Fedu": "father's education (0 none to 4 higher education)",
    "Mjob": "mother's job", "Fjob": "father's job",
    "reason": "reason for choosing this school",
    "guardian": "student's guardian",
    "traveltime": "home-to-school travel time (1 under 15 min to 4 over 1 h)",
    "studytime": "weekly study time (1 under 2 h to 4 over 10 h)",
    "failures": "number of past class failures (0-3)",
    "schoolsup": "extra educational support from the school", "famsup": "family educational support",
    "paid": "extra paid classes in the subject", "activities": "extra-curricular activities",
    "nursery": "attended nursery school", "higher": "wants to take higher education",
    "internet": "internet access at home", "romantic": "in a romantic relationship",
    "famrel": "quality of family relationships (1 very bad to 5 excellent)",
    "freetime": "free time after school (1-5)", "goout": "going out with friends (1-5)",
    "Dalc": "workday alcohol consumption (1-5)", "Walc": "weekend alcohol consumption (1-5)",
    "health": "current health status (1 very bad to 5 very good)",
    "absences": "number of school absences",
}

DROPOUT = {
    "Marital status": "marital status (coded)", "Application mode": "application mode (coded)",
    "Application order": "application order (0 first choice to 9 last)",
    "Course": "degree course (coded)",
    "Daytime/evening attendance": "daytime (1) or evening (0) attendance",
    "Previous qualification": "previous qualification (coded)",
    "Previous qualification (grade)": "grade of the previous qualification (0-200)",
    "Nacionality": "nationality (coded)",
    "Mother's qualification": "mother's qualification (coded)",
    "Father's qualification": "father's qualification (coded)",
    "Mother's occupation": "mother's occupation (coded)",
    "Father's occupation": "father's occupation (coded)",
    "Admission grade": "admission grade (0-200)", "Displaced": "displaced student",
    "Educational special needs": "educational special needs", "Debtor": "student is a debtor",
    "Tuition fees up to date": "tuition fees up to date", "Gender": "gender (1 male, 0 female)",
    "Scholarship holder": "scholarship holder", "Age at enrollment": "age at enrolment",
    "International": "international student",
    "Curricular units 1st sem (credited)": "curricular units credited in the first semester",
    "Curricular units 1st sem (enrolled)": "curricular units enrolled in the first semester",
    "Curricular units 1st sem (evaluations)": "evaluations to curricular units in the first semester",
    "Curricular units 1st sem (approved)": "curricular units approved in the first semester",
    "Curricular units 1st sem (grade)": "grade average in the first semester (0-20)",
    "Curricular units 1st sem (without evaluations)": "first-semester units without evaluations",
    "Curricular units 2nd sem (credited)": "curricular units credited in the second semester",
    "Curricular units 2nd sem (enrolled)": "curricular units enrolled in the second semester",
    "Curricular units 2nd sem (evaluations)": "evaluations to curricular units in the second semester",
    "Curricular units 2nd sem (approved)": "curricular units approved in the second semester",
    "Curricular units 2nd sem (grade)": "grade average in the second semester (0-20)",
    "Curricular units 2nd sem (without evaluations)": "second-semester units without evaluations",
    "Unemployment rate": "national unemployment rate at enrolment (%)",
    "Inflation rate": "national inflation rate at enrolment (%)", "GDP": "national GDP growth at enrolment",
}

SONAR = {f"band_{i:02d}": f"energy in frequency band {i} of 60 (ascending frequency), integrated "
         f"over a time window of the sonar chirp return" for i in range(1, 61)}

AMES = {
    "MS SubClass": "type of dwelling involved in the sale (coded)",
    "MS Zoning": "general zoning classification", "Lot Frontage": "linear feet of street connected to property",
    "Lot Area": "lot size in square feet", "Street": "type of road access", "Alley": "type of alley access",
    "Lot Shape": "general shape of property", "Land Contour": "flatness of the property",
    "Utilities": "type of utilities available", "Lot Config": "lot configuration",
    "Land Slope": "slope of property", "Neighborhood": "physical location within Ames city limits",
    "Condition 1": "proximity to main road or railroad", "Condition 2": "proximity to a second main road or railroad",
    "Bldg Type": "type of dwelling", "House Style": "style of dwelling",
    "Overall Qual": "overall material and finish quality (1-10)", "Overall Cond": "overall condition rating (1-10)",
    "Year Built": "original construction date", "Year Remod/Add": "remodel date",
    "Roof Style": "type of roof", "Roof Matl": "roof material",
    "Exterior 1st": "exterior covering on house", "Exterior 2nd": "second exterior covering",
    "Mas Vnr Type": "masonry veneer type", "Mas Vnr Area": "masonry veneer area in square feet",
    "Exter Qual": "exterior material quality", "Exter Cond": "present condition of the exterior material",
    "Foundation": "type of foundation", "Bsmt Qual": "height of the basement",
    "Bsmt Cond": "general condition of the basement", "Bsmt Exposure": "walkout or garden level basement walls",
    "BsmtFin Type 1": "quality of basement finished area", "BsmtFin SF 1": "type 1 finished square feet",
    "BsmtFin Type 2": "quality of second finished area", "BsmtFin SF 2": "type 2 finished square feet",
    "Bsmt Unf SF": "unfinished square feet of basement area", "Total Bsmt SF": "total square feet of basement area",
    "Heating": "type of heating", "Heating QC": "heating quality and condition",
    "Central Air": "central air conditioning", "Electrical": "electrical system",
    "1st Flr SF": "first floor square feet", "2nd Flr SF": "second floor square feet",
    "Low Qual Fin SF": "low quality finished square feet (all floors)",
    "Gr Liv Area": "above grade (ground) living area square feet",
    "Bsmt Full Bath": "basement full bathrooms", "Bsmt Half Bath": "basement half bathrooms",
    "Full Bath": "full bathrooms above grade", "Half Bath": "half baths above grade",
    "Bedroom AbvGr": "bedrooms above grade", "Kitchen AbvGr": "kitchens above grade",
    "Kitchen Qual": "kitchen quality", "TotRms AbvGrd": "total rooms above grade (excluding bathrooms)",
    "Functional": "home functionality rating", "Fireplaces": "number of fireplaces",
    "Fireplace Qu": "fireplace quality", "Garage Type": "garage location",
    "Garage Yr Blt": "year garage was built", "Garage Finish": "interior finish of the garage",
    "Garage Cars": "size of garage in car capacity", "Garage Area": "size of garage in square feet",
    "Garage Qual": "garage quality", "Garage Cond": "garage condition", "Paved Drive": "paved driveway",
    "Wood Deck SF": "wood deck area in square feet", "Open Porch SF": "open porch area in square feet",
    "Enclosed Porch": "enclosed porch area in square feet", "3Ssn Porch": "three season porch area in square feet",
    "Screen Porch": "screen porch area in square feet", "Pool Area": "pool area in square feet",
    "Pool QC": "pool quality", "Fence": "fence quality", "Misc Feature": "miscellaneous feature not covered elsewhere",
    "Misc Val": "value of miscellaneous feature in dollars", "Mo Sold": "month sold", "Yr Sold": "year sold",
    "Sale Type": "type of sale", "Sale Condition": "condition of sale",
}


# ------------------------------------------------------------ preparations

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    print("preparing data/benchmark/")

    red = pd.read_csv(RAW / "wine_quality/winequality-red.csv", sep=";")
    white = pd.read_csv(RAW / "wine_quality/winequality-white.csv", sep=";")
    red["colour"], white["colour"] = "red", "white"
    wine = pd.concat([red, white], ignore_index=True)
    wine = wine[[c for c in wine.columns if c != "quality"] + ["quality"]]
    write("wine_quality", wine, WINE)

    from sklearn.datasets import load_breast_cancer
    bc = load_breast_cancer(as_frame=True)
    df = bc.data.copy()
    df["diagnosis"] = np.where(bc.target.to_numpy() == 0, "M", "B")   # sklearn: 0 = malignant
    write("breast_cancer", df, WDBC)

    text = (RAW / "dermatology/dermatology.arff").read_text(encoding="utf-8", errors="replace")
    body = text.split("@data")[1] if "@data" in text else text.split("@DATA")[1]
    rows = [l.split(",") for l in body.strip().splitlines() if l.strip() and not l.startswith("%")]
    derm = pd.DataFrame(rows, columns=DERM_CLINICAL + DERM_HISTO + ["age", "class"])
    derm = derm.replace("?", np.nan)
    for c in derm.columns[:-1]:
        derm[c] = pd.to_numeric(derm[c])
    derm = median_impute(derm)
    names = {"1": "psoriasis", "2": "seboreic dermatitis", "3": "lichen planus", "4": "pityriasis rosea",
             "5": "cronic dermatitis", "6": "pityriasis rubra pilaris"}
    derm["class"] = derm["class"].str.strip().map(names)
    write("dermatology", derm, DERM)

    iono = pd.read_csv(RAW / "ionosphere/ionosphere.csv", header=None)
    iono.columns = [f"attribute_{i:02d}" for i in range(1, 35)] + ["class"]
    write("ionosphere", iono, IONO)

    arff = (RAW / "spambase/spambase.arff").read_text(encoding="utf-8", errors="replace")
    attrs = re.findall(r"@attribute\s+'?([^\s']+)'?", arff, flags=re.I)
    spam = pd.read_csv(RAW / "spambase/spambase.data", header=None)
    assert len(attrs) == spam.shape[1] == 58, (len(attrs), spam.shape)
    spam.columns = attrs[:-1] + ["class"]
    spam["class"] = np.where(spam["class"] == 1, "spam", "ham")
    write("spambase", spam, SPAM)

    with gzip.open(RAW / "optdigits/optdigits.tsv.gz", "rt") as fh:
        opt = pd.read_csv(io.StringIO(fh.read()), sep="\t")
    opt.columns = [f"pixel_{(i - 1) // 8}_{(i - 1) % 8}" for i in range(1, 65)] + ["digit"]
    opt["digit"] = opt["digit"].astype(int).astype(str)
    write("optdigits", opt, OPTD)

    geo = pd.read_csv(RAW / "geo_music/default_features_1059_tracks.txt", header=None)
    geo.columns = [f"feature_{i:02d}" for i in range(1, 69)] + ["latitude", "longitude"]
    geo = geo[[c for c in geo.columns if c != "latitude"] + ["latitude"]]
    write("geo_music", geo, {**GEO, "longitude": "longitude of the region of origin (dropped before use)"})

    mice = pd.read_csv(RAW / "mice_protein/mice.csv", encoding="utf-8-sig")
    mice = median_impute(mice)
    mice = mice[[c for c in mice.columns if c != "class"] + ["class"]]
    write("mice_protein", mice, {**MICE, "MouseID": "mouse identifier (dropped before use)",
                                 "Genotype": "control or trisomic (defines the class; dropped before use)",
                                 "Treatment": "memantine or saline (defines the class; dropped before use)",
                                 "Behavior": "context-shock or shock-context (defines the class; dropped before use)"})

    park = pd.read_csv(RAW / "parkinsons/parkinsons.csv")
    park = park[[c for c in park.columns if c != "total_UPDRS"] + ["total_UPDRS"]]
    write("parkinsons", park, {**PARK, "subject#": "subject identifier (dropped before use)",
                               "motor_UPDRS": "motor UPDRS score (alternative target; dropped before use)"})

    por = pd.read_csv(RAW / "student_performance/student-por.csv", sep=";")
    write("student_performance", por, {**STUDENT, "G1": "first period grade (dropped before use)",
                                       "G2": "second period grade (dropped before use)"})

    drop = pd.read_csv(RAW / "student_dropout/data.csv", sep=";", encoding="utf-8-sig")
    drop.columns = [c.strip() for c in drop.columns]
    write("student_dropout", drop, DROPOUT)

    sonar = pd.read_csv(RAW / "sonar/sonar.csv", header=None)
    sonar.columns = [f"band_{i:02d}" for i in range(1, 61)] + ["class"]
    write("sonar", sonar, SONAR)

    ames = pd.read_csv(RAW / "ames_housing/housing.csv")
    ames = ames.drop(columns=[c for c in ames.columns if c.startswith("Unnamed")])
    for c in ames.columns:
        if not pd.api.types.is_numeric_dtype(ames[c]):
            ames[c] = ames[c].fillna("Absent")
    ames = median_impute(ames)
    ames["SalePrice"] = np.log(ames["SalePrice"].astype(float))
    ames = ames[[c for c in ames.columns if c != "SalePrice"] + ["SalePrice"]]
    write("ames_housing", ames, {**AMES, "Order": "observation number (dropped before use)",
                                 "PID": "parcel identification number (dropped before use)"})


def main():
    if "--prepare" not in sys.argv:
        download()
    prepare()


if __name__ == "__main__":
    main()
