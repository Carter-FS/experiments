import pandas as pd

clinical_factor_columns = [
    "sex",
    "age",
    "pretrt_sz_5",
    "focal",
    "fam_hx",
    "febrile",
    "ci",
    "birth_t",
    "head",
    "drug",
    "alcohol",
    "cvd",
    "psy",
    "ld",
    "lesion",
    "eeg_cat",
]


def group_age(age):
    if age <= 4:
        return age
    elif age < 18:
        return 1
    elif 18 <= age <= 29:
        return 2
    elif 29 < age <= 46:
        return 3
    elif age > 46:
        return 4

        
def process_clinifical_factors(x):
    for col in clinical_factor_columns:
        if pd.isna(x[col]):
            x[col] = 0
            continue

        if col == "age":
            x[col] = group_age(x[col])
            continue

        if col == "pretrt_sz_5":
            if x[col] in ["<=5", 0]:
                x[col] = 1
            elif x[col] in [">5", 1]:
                x[col] = 2
            else:
                x[col] = 0
            continue

        if col == "sex":
            if x[col] in ["M", "Male", 1]:
                x[col] = 2
            elif x[col] in ["F", "Female", 0]:
                x[col] = 1
            else:
                x[col] = 0
            continue

        if col in ["lesion", "lesion_cat"]:
            lesion_map = {
              # Missing/unavailable
              "not available": 0,
              0: 0,
              "0": 0,
              "0.0": 0,
            
              # Normal
              "n": 1,
              "normal": 1,
              1: 1,
              "1": 1,
              "1.0": 1,
            
              # Abnormal
              "a": 2,
              "abnormal": 2,
              2: 2,
              "2": 2,
              "2.0": 2,
            
              # Epileptiogenic
              "e": 3,
              "epileptiogenic": 3,  # spelling present in the data
              "epileptogenic": 3,   # alternative spelling
              3: 3,
              "3": 3,
              "3.0": 3,
            }
 
            x[col] = lesion_map.get(x[col], 0)
            continue

        if col == "eeg_cat":
            if x[col] in ["Epileptiform", "E"]:
                x[col] = 3
            elif x[col] in ["Normal", "N"]:
                x[col] = 1
            elif x[col] in ["Abnormal", "A"]:
                x[col] = 2
            else:
                x[col] = 0
            continue

        if x[col] in [1, "Yes"]:
            x[col] = 2
        elif x[col] == 2:
            x[col] = 0
        elif x[col] in [0, "No"]:
            x[col] = 1
        else:
            x[col] = 0
    return x