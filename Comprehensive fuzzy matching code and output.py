#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# Script: fuzzy_match_TRI_Preqin_progress.py
# Purpose:
#   1. Load the cleaned TRI dataset and the cleaned Preqin dataset.
#   2. Standardise key text fields (e.g., facility names in TRI, firm names in Preqin).
#   3. Use fuzzy matching with a progress bar to match TRI facility names 
#      with Preqin firm names.
#   4. Save the matched results for further analysis.
#
# Author: Roy Adefarakan
# Date: [Insert Date]
# =============================================================================

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rapidfuzz import process, fuzz
from tqdm import tqdm  # Progress bar

# =============================================================================
# 1. Load the Cleaned Datasets
# =============================================================================

# File paths (update paths as needed)
tri_file = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\TRI_Combined_Cleaned.csv"
preqin_file = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\Preqin_Cleaned.xlsx"

# Load TRI dataset
try:
    tri_df = pd.read_csv(tri_file, low_memory=False)
    print("Loaded TRI dataset with shape:", tri_df.shape)
except Exception as e:
    print("Error loading TRI dataset:", e)
    raise

# Load Preqin dataset
try:
    preqin_df = pd.read_excel(preqin_file)
    print("Loaded Preqin dataset with shape:", preqin_df.shape)
except Exception as e:
    print("Error loading Preqin dataset:", e)
    raise

# =============================================================================
# 2. Standardise Column Names and Key Text Fields
# =============================================================================

# --- For TRI dataset ---
tri_df.columns = [col.strip().lower().replace(' ', '_') for col in tri_df.columns]
print("TRI columns after standardisation:")
print(tri_df.columns.tolist())

# Ensure TRI has a 'facility_name' column.
if 'facility_name' not in tri_df.columns:
    # Look for alternative column that contains both "facility" and "name"
    alt_col = None
    for col in tri_df.columns:
        if re.search(r'facility.*name', col, re.IGNORECASE):
            alt_col = col
            break
    if alt_col:
        print(f"Renaming column '{alt_col}' to 'facility_name'")
        tri_df.rename(columns={alt_col: 'facility_name'}, inplace=True)
    else:
        raise KeyError("Error: 'facility_name' column not found in TRI dataset.")
else:
    print("'facility_name' column exists in TRI dataset.")

# Clean the 'facility_name' field in TRI dataset.
tri_df['facility_name'] = tri_df['facility_name'].astype(str).str.strip().str.lower()
tri_df['facility_name'] = tri_df['facility_name'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# --- For Preqin dataset ---
preqin_df.columns = [col.strip().lower().replace(' ', '_') for col in preqin_df.columns]
print("Preqin columns after standardisation:")
print(preqin_df.columns.tolist())

# Identify the firm identifier column in Preqin.
if 'target_company' in preqin_df.columns:
    preqin_df['target_company'] = preqin_df['target_company'].astype(str).str.strip().str.lower()
    firm_identifier = 'target_company'
    print("Using 'target_company' as firm identifier in Preqin.")
elif 'firm_name' in preqin_df.columns:
    preqin_df['firm_name'] = preqin_df['firm_name'].astype(str).str.strip().str.lower()
    firm_identifier = 'firm_name'
    print("Using 'firm_name' as firm identifier in Preqin.")
else:
    raise KeyError("Error: Neither 'target_company' nor 'firm_name' found in Preqin dataset.")

# Create a list of unique firm names from Preqin for fuzzy matching.
preqin_firms = preqin_df[firm_identifier].unique().tolist()
print("Number of unique firm names in Preqin:", len(preqin_firms))

# =============================================================================
# 3. Define Fuzzy Matching Function Using Levenshtein Distance
# =============================================================================

def fuzzy_match_facility(facility_name, choices, scorer=fuzz.token_sort_ratio, threshold=80):
    """
    Uses rapidfuzz to fuzzy match a facility name against a list of choices.
    
    Parameters:
      facility_name (str): The facility name from TRI.
      choices (list): List of firm names from Preqin.
      scorer (function): Scoring function from rapidfuzz (default: fuzz.token_sort_ratio).
      threshold (int): Minimum similarity score (0-100) to consider a match valid.
    
    Returns:
      (matched_name, match_score): Tuple with the best matching firm name and its score, or (None, None) if no match is found.
    """
    try:
        match = process.extractOne(facility_name, choices, scorer=scorer)
        if match and match[1] >= threshold:
            return match[0], match[1]
        else:
            return None, None
    except Exception as e:
        print(f"Error matching '{facility_name}':", e)
        return None, None

# =============================================================================
# 4. Apply Fuzzy Matching with Progress Bar
# =============================================================================

# Initialize new columns for matched firm and match score
tri_df['matched_firm'] = None
tri_df['match_score'] = None

# Use tqdm to track progress over facility names
print("Starting fuzzy matching on TRI facility names...")
for idx, facility in tqdm(enumerate(tri_df['facility_name']), total=len(tri_df)):
    match_name, score = fuzzy_match_facility(facility, preqin_firms, threshold=80)
    tri_df.at[idx, 'matched_firm'] = match_name
    tri_df.at[idx, 'match_score'] = score

# Sample output for verification
print("Sample fuzzy matching results:")
print(tri_df[['facility_name', 'matched_firm', 'match_score']].head(20))

# =============================================================================
# 5. Visual Diagnostics of Fuzzy Matching
# =============================================================================

plt.figure(figsize=(8, 5))
plt.hist(tri_df['match_score'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Fuzzy Match Scores')
plt.xlabel('Match Score')
plt.ylabel('Frequency')
plt.savefig("fuzzy_matching_distribution.png")
plt.show()

unique_matched_firms = tri_df['matched_firm'].nunique()
print("Number of unique matched firms in TRI data:", unique_matched_firms)

# =============================================================================
# 6. Aggregate TRI Data to Firm-Level
# =============================================================================

# Define which emission columns to aggregate.
# If there is a specific column 'total_emission', use it; otherwise, use all columns with 'emission' in the name.
if 'total_emission' in tri_df.columns:
    emission_agg_cols = ['total_emission']
else:
    emission_agg_cols = [col for col in tri_df.columns if 'emission' in col]

# Group by the matched firm name and aggregate (sum) emission values.
firm_emissions = tri_df.groupby('matched_firm')[emission_agg_cols].sum().reset_index()
print("Aggregated firm-level emissions shape:", firm_emissions.shape)

# =============================================================================
# 7. Save the Final Matched and Aggregated Dataset
# =============================================================================

output_matched_path = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\TRI_Preqin_Matched_Aggregated.csv"
firm_emissions.to_csv(output_matched_path, index=False)
print("Matched and aggregated firm-level emissions data saved to:", output_matched_path)


# In[2]:


# =============================================================================
# Script: merge_TRI_Preqin_then_AMPD.py
# Purpose:
#   1. Load the cleaned TRI and Preqin datasets.
#   2. Standardise key text fields.
#   3. Perform hierarchical fuzzy matching between TRI facility names and 
#      Preqin firm names using state/ZIP filters and advanced text preprocessing.
#   4. Merge all columns from TRI and Preqin into a single DataFrame.
#   5. (Optional) Merge the result with the AMPD dataset.
#
# Author: [Your Name]
# Date: [Insert Date]
# =============================================================================

import os
import re
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Load Cleaned Datasets using Provided File Paths
# -----------------------------------------------------------------------------

# File paths
tri_file = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\TRI_Combined_Cleaned.csv"
preqin_file = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\Preqin_Cleaned.xlsx"
ampd_file = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\AMPD_Annual_Emission_1995_2023_Cleaned.csv"  # Optional

# Load TRI dataset
tri_df = pd.read_csv(tri_file, low_memory=False)
print("Loaded TRI dataset with shape:", tri_df.shape)

# Load Preqin dataset
preqin_df = pd.read_excel(preqin_file)
print("Loaded Preqin dataset with shape:", preqin_df.shape)

# (Optional) Load AMPD dataset
try:
    ampd_df = pd.read_csv(ampd_file, low_memory=False)
    print("Loaded AMPD dataset with shape:", ampd_df.shape)
except Exception as e:
    print("AMPD dataset not loaded:", e)
    ampd_df = None

# -----------------------------------------------------------------------------
# 2. Standardise Column Names & Identify Key Columns
# -----------------------------------------------------------------------------

def standardise_columns(df):
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

tri_df = standardise_columns(tri_df)
preqin_df = standardise_columns(preqin_df)
if ampd_df is not None:
    ampd_df = standardise_columns(ampd_df)

print("TRI columns after standardisation:")
print(tri_df.columns.tolist())
print("Preqin columns after standardisation:")
print(preqin_df.columns.tolist())

# Ensure TRI has a 'facility_name' column; if not, try to detect an alternative
if 'facility_name' not in tri_df.columns:
    alt_col = None
    for col in tri_df.columns:
        if re.search(r'facility.*name', col, re.IGNORECASE):
            alt_col = col
            break
    if alt_col:
        print(f"Renaming column '{alt_col}' to 'facility_name'")
        tri_df.rename(columns={alt_col: 'facility_name'}, inplace=True)
    else:
        raise KeyError("Error: 'facility_name' column not found in TRI dataset.")
else:
    print("'facility_name' column exists in TRI dataset.")

# For Preqin, identify the firm identifier column.
if 'target_company' in preqin_df.columns:
    firm_identifier = 'target_company'
    print("Using 'target_company' as firm identifier in Preqin.")
elif 'firm_name' in preqin_df.columns:
    firm_identifier = 'firm_name'
    print("Using 'firm_name' as firm identifier in Preqin.")
else:
    raise KeyError("Error: Neither 'target_company' nor 'firm_name' found in Preqin dataset.")

# -----------------------------------------------------------------------------
# 3. Advanced String Preprocessing
# -----------------------------------------------------------------------------

# Define synonyms and stopwords for enhanced cleaning
SYNONYMS = {
    "inc": "incorporated",
    "co": "company",
    "corp": "corporation",
    "mfg": "manufacturing",
    # Extend as needed
}
STOPWORDS = {"the", "of", "llc", "ltd", "and", "plant", "main", "facility"}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    tokens = [SYNONYMS.get(token, token) for token in tokens if token not in STOPWORDS]
    return " ".join(tokens)

# Clean facility_name in TRI
tri_df['facility_name'] = tri_df['facility_name'].apply(clean_text)
# Clean Preqin firm identifier column
preqin_df[firm_identifier] = preqin_df[firm_identifier].apply(clean_text)

# Create a list of unique firm names from Preqin
preqin_firms = preqin_df[firm_identifier].unique().tolist()
print("Number of unique firm names in Preqin:", len(preqin_firms))

# -----------------------------------------------------------------------------
# 4. Hierarchical Fuzzy Matching Function (with State/ZIP Filtering)
# -----------------------------------------------------------------------------

def hierarchical_match(tri_row, preqin_list, threshold_main=80, threshold_lower=75):
    """
    Hierarchical fuzzy matching using additional metadata.
    
    Steps:
      1. (Optional) Filter by exact matching on state or ZIP if available.
      2. Apply fuzzy matching on facility_name within the filtered subset.
      3. If the best match is >= threshold_main, accept it.
      4. If the score is between threshold_lower and threshold_main, flag as 'needs_review'.
    
    Returns:
      (matched_name, score, needs_review)
    """
    facility_name = tri_row['facility_name']
    # For demonstration: here, we are not filtering by state/zip because Preqin may not have these.
    # In a full implementation, you could subset preqin_list by tri_row['state'] or tri_row['zip'] if available.
    
    # Fuzzy matching on facility name
    match = process.extractOne(facility_name, preqin_list, scorer=fuzz.token_sort_ratio)
    if match:
        best_match, score = match[0], match[1]
        needs_review = False
        if score >= threshold_main:
            return best_match, score, needs_review
        elif score >= threshold_lower:
            needs_review = True
            return best_match, score, needs_review
    return None, None, False

# -----------------------------------------------------------------------------
# 5. Apply Hierarchical Fuzzy Matching with Progress Bar
# -----------------------------------------------------------------------------

# Create columns for the cleaned TRI facility name, matched firm name, match score, and review flag
tri_df['clean_name_tri'] = tri_df['facility_name']  # Store cleaned facility name
tri_df['matched_name'] = None
tri_df['match_score'] = None
tri_df['needs_review'] = False

print("Starting hierarchical fuzzy matching with progress bar...")

for idx in tqdm(range(len(tri_df)), desc="Fuzzy Matching"):
    row = tri_df.iloc[idx]
    best_match, score, review = hierarchical_match(row, preqin_firms, threshold_main=80, threshold_lower=75)
    tri_df.at[idx, 'matched_name'] = best_match
    tri_df.at[idx, 'match_score'] = score
    tri_df.at[idx, 'needs_review'] = review

# Optional: Create a score group column for further analysis
def score_group(score):
    if pd.isnull(score):
        return 'no_match'
    elif score >= 90:
        return '90-100'
    elif score >= 80:
        return '80-89'
    else:
        return 'below80'

tri_df['score_group'] = tri_df['match_score'].apply(score_group)

# -----------------------------------------------------------------------------
# 6. Visual Diagnostics of Fuzzy Matching
# -----------------------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(tri_df['match_score'].dropna(), bins=20, kde=True, color='skyblue')
plt.title('Hierarchical Fuzzy Match Score Distribution')
plt.xlabel('Match Score')
plt.ylabel('Frequency')
plt.savefig("hierarchical_fuzzy_score_distribution.png")
plt.show()

unique_matches = tri_df['matched_name'].nunique()
print("Number of unique matched firm names in TRI:", unique_matches)

# -----------------------------------------------------------------------------
# 7. Merge with Preqin Data to Create a Unified DataFrame
# -----------------------------------------------------------------------------

# Rename matched_name to 'clean_name_preqin' in TRI for consistency with Preqin
tri_df.rename(columns={'matched_name': 'clean_name_preqin'}, inplace=True)

# Also rename the Preqin firm identifier column to 'clean_name_preqin'
preqin_df.rename(columns={firm_identifier: 'clean_name_preqin'}, inplace=True)

# Merge TRI and Preqin using a left join on 'clean_name_preqin'
merged_df = pd.merge(tri_df, preqin_df, how='left', on='clean_name_preqin', suffixes=('_tri', '_preqin'))
print("Merged TRI-Preqin dataset shape:", merged_df.shape)

# -----------------------------------------------------------------------------
# 8. (Optional) Merge with AMPD Dataset
# -----------------------------------------------------------------------------

if ampd_df is not None:
    # Standardise AMPD facility_name, state, zip columns similarly
    ampd_df = standardise_columns(ampd_df)
    if 'facility_name' not in ampd_df.columns:
        # Attempt to find a similar column in AMPD
        alt_ampd = None
        for col in ampd_df.columns:
            if re.search(r'facility.*name', col, re.IGNORECASE):
                alt_ampd = col
                break
        if alt_ampd:
            ampd_df.rename(columns={alt_ampd: 'facility_name'}, inplace=True)
        else:
            raise KeyError("Error: 'facility_name' column not found in AMPD dataset.")
    ampd_df['facility_name'] = ampd_df['facility_name'].astype(str).apply(clean_text)
    # Optionally, you can perform a similar fuzzy matching procedure for AMPD
    # For now, we'll merge AMPD based on a common firm identifier if available.
    
    # Assume AMPD has a column 'firm_id' that corresponds to Preqin's firm id. If not, you can use similar fuzzy matching.
    # Here, we perform a left join between merged_df and ampd_df based on 'clean_name_preqin'
    merged_df = pd.merge(merged_df, ampd_df, how='left', on='clean_name_preqin', suffixes=('', '_ampd'))
    print("Merged TRI-Preqin-AMPD dataset shape:", merged_df.shape)

# -----------------------------------------------------------------------------
# 9. Save the Final Comprehensive Dataset
# -----------------------------------------------------------------------------

output_path = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\TRI_Preqin_Comprehensive_Matched.csv"
merged_df.to_csv(output_path, index=False)
print("Final comprehensive dataset saved to:", output_path)


# In[3]:


# =============================================================================
# Script: hierarchical_fuzzy_match_with_ampd.py
# Purpose:
#   1. Load the cleaned TRI, Preqin, and (optionally) AMPD datasets.
#   2. Standardise key text fields and columns.
#   3. Perform hierarchical fuzzy matching between TRI facility names and 
#      Preqin firm names, using additional metadata (e.g., state, ZIP) if available.
#   4. Use dynamic thresholding to flag borderline matches.
#   5. Merge the matched TRI data with Preqin data (and optionally AMPD).
#   6. Save the final comprehensive master dataset.
#
# Author: [Your Name]
# Date: [Insert Date]
# =============================================================================

import os
import re
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Load Cleaned Datasets (using provided file paths)
# -----------------------------------------------------------------------------

tri_file = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\TRI_Combined_Cleaned.csv"
preqin_file = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\Preqin_Cleaned.xlsx"
ampd_file = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\AMPD_Annual_Emission_1995_2023_Cleaned.csv"  # Optional

# Load TRI dataset
tri_df = pd.read_csv(tri_file, low_memory=False)
print("Loaded TRI dataset with shape:", tri_df.shape)

# Load Preqin dataset
preqin_df = pd.read_excel(preqin_file)
print("Loaded Preqin dataset with shape:", preqin_df.shape)

# Optional: Load AMPD dataset
try:
    ampd_df = pd.read_csv(ampd_file, low_memory=False)
    print("Loaded AMPD dataset with shape:", ampd_df.shape)
except Exception as e:
    print("AMPD dataset not loaded:", e)
    ampd_df = None

# -----------------------------------------------------------------------------
# 2. Standardise Column Names & Identify Key Columns
# -----------------------------------------------------------------------------

def standardise_columns(df):
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

tri_df = standardise_columns(tri_df)
preqin_df = standardise_columns(preqin_df)
if ampd_df is not None:
    ampd_df = standardise_columns(ampd_df)

print("TRI columns after standardisation:")
print(tri_df.columns.tolist())
print("Preqin columns after standardisation:")
print(preqin_df.columns.tolist())

# Ensure TRI has a 'facility_name' column; if not, try to find an alternative
if 'facility_name' not in tri_df.columns:
    alt_col = None
    for col in tri_df.columns:
        if re.search(r'facility.*name', col, re.IGNORECASE):
            alt_col = col
            break
    if alt_col:
        print(f"Renaming column '{alt_col}' to 'facility_name'")
        tri_df.rename(columns={alt_col: 'facility_name'}, inplace=True)
    else:
        raise KeyError("Error: 'facility_name' column not found in TRI dataset.")
else:
    print("'facility_name' column exists in TRI dataset.")

# For Preqin, identify the firm identifier column (prefer 'target_company')
if 'target_company' in preqin_df.columns:
    firm_identifier = 'target_company'
    print("Using 'target_company' as firm identifier in Preqin.")
elif 'firm_name' in preqin_df.columns:
    firm_identifier = 'firm_name'
    print("Using 'firm_name' as firm identifier in Preqin.")
else:
    raise KeyError("Error: Neither 'target_company' nor 'firm_name' found in Preqin dataset.")

# -----------------------------------------------------------------------------
# 3. Advanced String Preprocessing: Synonyms, Stopwords, etc.
# -----------------------------------------------------------------------------

# Define synonyms and stopwords for enhanced text cleaning
SYNONYMS = {
    "inc": "incorporated",
    "co": "company",
    "corp": "corporation",
    "mfg": "manufacturing",
    # Extend as needed
}
STOPWORDS = {"the", "of", "llc", "ltd", "and", "plant", "main", "facility"}

def clean_text(text):
    """Clean text by lowercasing, removing punctuation, replacing synonyms, and removing stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    cleaned_tokens = [SYNONYMS.get(token, token) for token in tokens if token not in STOPWORDS]
    return " ".join(cleaned_tokens)

# Clean TRI facility names
tri_df['facility_name'] = tri_df['facility_name'].apply(clean_text)

# Clean Preqin firm names
preqin_df[firm_identifier] = preqin_df[firm_identifier].apply(clean_text)

# Optionally, if available in TRI, clean state and zip fields for hierarchical matching
if 'state' in tri_df.columns:
    tri_df['state'] = tri_df['state'].astype(str).apply(clean_text)
if 'zip' in tri_df.columns:
    tri_df['zip'] = tri_df['zip'].astype(str).apply(clean_text)

# Create a list of unique Preqin firm names
preqin_firms = preqin_df[firm_identifier].unique().tolist()
print("Number of unique firm names in Preqin:", len(preqin_firms))

# -----------------------------------------------------------------------------
# 4. Hierarchical Fuzzy Matching Function (with State/ZIP Filtering)
# -----------------------------------------------------------------------------

def hierarchical_match(tri_row, preqin_list, threshold_main=80, threshold_lower=75):
    """
    Hierarchical fuzzy matching with optional filtering by state/ZIP.
    
    For demonstration, this function performs:
      1. Fuzzy matching on the 'facility_name' of the TRI row against the full preqin_list.
      2. Returns (best_match, score, needs_review) based on dynamic thresholds.
    
    Parameters:
      tri_row (pd.Series): A row from the TRI DataFrame.
      preqin_list (list): List of preprocessed firm names from Preqin.
      threshold_main (int): Main threshold for automatic acceptance.
      threshold_lower (int): Lower threshold for borderline matches.
    
    Returns:
      (matched_name, score, needs_review)
    """
    facility_name = tri_row['facility_name']
    
    # (Optional) Here, if Preqin had state/zip information, you could filter the preqin_list accordingly.
    # For now, we perform matching on the entire list.
    match = process.extractOne(facility_name, preqin_list, scorer=fuzz.token_sort_ratio)
    if match:
        best_match, score = match[0], match[1]
        needs_review = False
        if score >= threshold_main:
            return best_match, score, needs_review
        elif score >= threshold_lower:
            needs_review = True
            return best_match, score, needs_review
    return None, None, False

# -----------------------------------------------------------------------------
# 5. Apply Hierarchical Fuzzy Matching with Progress Bar
# -----------------------------------------------------------------------------

# Create new columns for the cleaned TRI name, matched firm name, match score, and review flag.
tri_df['clean_name_tri'] = tri_df['facility_name']  # store the cleaned facility name
tri_df['matched_name'] = None
tri_df['match_score'] = None
tri_df['needs_review'] = False

print("Starting hierarchical fuzzy matching on TRI facility names...")

# Apply the hierarchical_match function with a progress bar
for idx in tqdm(range(len(tri_df)), desc="Fuzzy Matching"):
    row = tri_df.iloc[idx]
    best_match, score, review = hierarchical_match(row, preqin_firms, threshold_main=80, threshold_lower=75)
    tri_df.at[idx, 'matched_name'] = best_match
    tri_df.at[idx, 'match_score'] = score
    tri_df.at[idx, 'needs_review'] = review

# Optional: Create a score group column for additional analysis.
def score_group(score):
    if pd.isnull(score):
        return 'no_match'
    elif score >= 90:
        return '90-100'
    elif score >= 80:
        return '80-89'
    else:
        return 'below80'
tri_df['score_group'] = tri_df['match_score'].apply(score_group)

# Display sample fuzzy matching results
print("Sample fuzzy matching results:")
print(tri_df[['clean_name_tri', 'matched_name', 'match_score', 'score_group', 'needs_review']].head(20))

# -----------------------------------------------------------------------------
# 6. Visual Diagnostics of Fuzzy Matching
# -----------------------------------------------------------------------------

plt.figure(figsize=(8, 5))
try:
    sns.histplot(tri_df['match_score'].dropna(), bins=20, kde=True, color='skyblue')
except AttributeError:
    # For older versions of seaborn
    sns.distplot(tri_df['match_score'].dropna(), bins=20, kde=True, color='skyblue')
plt.title('Hierarchical Fuzzy Match Score Distribution')
plt.xlabel('Match Score')
plt.ylabel('Frequency')
plt.savefig("hierarchical_fuzzy_score_distribution.png")
plt.show()

unique_matches = tri_df['matched_name'].nunique()
print("Number of unique matched firm names in TRI:", unique_matches)

# -----------------------------------------------------------------------------
# 7. Merge TRI & Preqin Data to Create a Master DataFrame
# -----------------------------------------------------------------------------

# Rename TRI column 'matched_name' to 'clean_name_preqin' for consistency
tri_df.rename(columns={'matched_name': 'clean_name_preqin'}, inplace=True)

# Rename Preqin firm identifier column to 'clean_name_preqin'
preqin_df.rename(columns={firm_identifier: 'clean_name_preqin'}, inplace=True)

# Merge TRI and Preqin on 'clean_name_preqin' using a left join.
merged_df = pd.merge(tri_df, preqin_df, how='left', on='clean_name_preqin', suffixes=('_tri', '_preqin'))
print("Merged TRI-Preqin dataset shape:", merged_df.shape)

# -----------------------------------------------------------------------------
# 8. (Optional) Merge with AMPD Data
# -----------------------------------------------------------------------------

if ampd_df is not None:
    # Standardise AMPD facility names similar to TRI
    if 'facility_name' not in ampd_df.columns:
        alt_ampd = None
        for col in ampd_df.columns:
            if re.search(r'facility.*name', col, re.IGNORECASE):
                alt_ampd = col
                break
        if alt_ampd:
            ampd_df.rename(columns={alt_ampd: 'facility_name'}, inplace=True)
        else:
            raise KeyError("Error: 'facility_name' column not found in AMPD dataset.")
    ampd_df['facility_name'] = ampd_df['facility_name'].astype(str).apply(clean_text)
    
    # Optionally perform fuzzy matching for AMPD similar to TRI
    # For now, assume AMPD has a direct firm identifier column, or repeat fuzzy matching
    # Here, we'll merge on 'clean_name_preqin' if available in AMPD; otherwise, skip
    if 'clean_name_preqin' not in ampd_df.columns:
        # Optionally, if AMPD contains facility names, perform a simple fuzzy match on those as well.
        ampd_df['clean_name_preqin'] = ampd_df['facility_name']  # For demonstration; ideally use fuzzy matching
    merged_df = pd.merge(merged_df, ampd_df, how='left', on='clean_name_preqin', suffixes=('', '_ampd'))
    print("Merged TRI-Preqin-AMPD dataset shape:", merged_df.shape)

# -----------------------------------------------------------------------------
# 9. Save Final Master Dataset
# -----------------------------------------------------------------------------

output_path = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\TRI_Preqin_Comprehensive_Matched.csv"
merged_df.to_csv(output_path, index=False)
print("Final comprehensive dataset saved to:", output_path)


# In[1]:


# =============================================================================
# Script: analyze_merged_data.py
# Purpose:
#   1. Load the final merged dataset (TRI, Preqin, AMPD).
#   2. Inspect matching results (match_score, needs_review).
#   3. Perform basic descriptive analyses (missing values, summary stats).
#   4. Demonstrate how to create a firm-year panel for further econometric analysis.
#
# Author: [Your Name]
# Date: [Insert Date]
# =============================================================================

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Load the Merged Dataset
# -----------------------------------------------------------------------------

merged_file = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\TRI_Preqin_Comprehensive_Matched.csv"

try:
    df = pd.read_csv(merged_file, low_memory=False)
    print(f"Loaded merged dataset with shape: {df.shape}")
except Exception as e:
    print("Error loading merged dataset:", e)
    df = pd.DataFrame()

# -----------------------------------------------------------------------------
# 2. Inspect Basic Structure
# -----------------------------------------------------------------------------

print("Column names:\n", df.columns.tolist())
print("\nSample rows:")
print(df.head(10))

# Check how many rows have a non-null match_score
if 'match_score' in df.columns:
    matched_count = df['match_score'].notnull().sum()
    total_count = len(df)
    print(f"\nNumber of matched rows: {matched_count} out of {total_count} ({matched_count/total_count:.2%})")
else:
    print("Warning: 'match_score' column not found. Did the fuzzy matching script produce it?")

# If 'needs_review' was created, see how many are flagged
if 'needs_review' in df.columns:
    review_count = df['needs_review'].sum()
    print(f"Number of rows flagged for review: {review_count}")
else:
    print("No 'needs_review' column found.")

# -----------------------------------------------------------------------------
# 3. Examine AMPD Columns
# -----------------------------------------------------------------------------

# If AMPD columns were merged, they might have a suffix like "_ampd" or certain known names.
ampd_cols = [col for col in df.columns if '_ampd' in col or 'ampd' in col.lower()]
print(f"\nPotential AMPD columns found: {ampd_cols}")

# Check how many rows have non-null values in these columns
for col in ampd_cols:
    non_null_count = df[col].notnull().sum()
    print(f"Column '{col}' non-null count: {non_null_count}")

# -----------------------------------------------------------------------------
# 4. Descriptive Statistics
# -----------------------------------------------------------------------------

# Identify a few numeric columns of interest (e.g., 'total_releases' from TRI, 
# or any AMPD emission columns)
candidate_cols = []
if '107._total_releases' in df.columns:
    candidate_cols.append('107._total_releases')
for col in ampd_cols:
    # If you have known numeric emission columns from AMPD, add them here
    pass

print("\nDescriptive statistics for selected numeric columns:")
print(df[candidate_cols].describe())

# Visualise distribution of 'match_score' if present
if 'match_score' in df.columns:
    plt.figure(figsize=(8, 5))
    valid_scores = df['match_score'].dropna()
    if not valid_scores.empty:
        try:
            sns.histplot(valid_scores, bins=20, kde=True, color='skyblue')
        except AttributeError:
            sns.distplot(valid_scores, bins=20, kde=True, color='skyblue')
        plt.title("Distribution of Match Scores in Merged Data")
        plt.xlabel("Match Score")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("No non-null match_score values to plot.")

# -----------------------------------------------------------------------------
# 5. Construct a Simple Firm-Year Panel (Example)
# -----------------------------------------------------------------------------

# Suppose we want to create a firm-year dataset from the TRI data, 
# aggregated by 'clean_name_preqin' and 'year'.

if 'clean_name_preqin' in df.columns and 'year' in df.columns:
    # Example: sum total releases per (firm, year)
    panel_df = (df.groupby(['clean_name_preqin', 'year'])['107._total_releases']
                  .sum()
                  .reset_index())
    panel_df.rename(columns={'107._total_releases': 'total_releases_sum'}, inplace=True)
    print("\nPanel DataFrame shape:", panel_df.shape)
    print(panel_df.head(10))
    
    # Merge panel_df with Preqin columns if needed
    # e.g. if Preqin has 'deal_date' or 'deal_id' that can be joined by year or something
    # This depends on your analysis plan. For demonstration:
    #   merged_panel = pd.merge(panel_df, df[['clean_name_preqin','deal_id','year']], 
    #                           on=['clean_name_preqin','year'], how='left')
    #   ...
else:
    print("\nCannot construct firm-year panel: 'clean_name_preqin' or 'year' missing.")

# -----------------------------------------------------------------------------
# 6. Potential Next Steps
# -----------------------------------------------------------------------------
#  - Outlier detection in emissions data
#  - Imputation or dropping of missing values
#  - Creating a 'treatment' variable for DiD if analyzing the effect of PE acquisitions
#  - Rolling regressions or event studies
#  - Additional merges with governance or financial data

print("\nAnalysis script completed. You may proceed with deeper econometric modeling.")


# In[3]:


import os
import csv
from tqdm import tqdm

def split_csv(input_file, output_folder, output_prefix, lines_per_file=200000, encoding='utf-8'):
    """
    Splits a large CSV file into multiple smaller CSV files.
    
    Parameters:
      input_file (str): Path to the large CSV file.
      output_folder (str): Folder where the smaller CSV files will be saved.
      output_prefix (str): Prefix for naming the output files.
      lines_per_file (int): Maximum number of data rows (excluding header) per chunk.
      encoding (str): File encoding for reading/writing.
    
    The function writes each chunk as a new CSV file with the same header as the input.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the large input CSV file for reading
    with open(input_file, 'r', newline='', encoding=encoding) as f_in:
        reader = csv.reader(f_in)
        header = next(reader)  # read header
        
        # Count total lines for progress bar estimation (optional, but can be slow for huge files)
        total_lines = sum(1 for row in f_in)
        f_in.seek(0)
        next(reader)  # Skip header again
        
        # Initialize variables for chunking
        file_count = 1
        current_line = 0
        output_file = os.path.join(output_folder, f"{output_prefix}_part{file_count}.csv")
        f_out = open(output_file, 'w', newline='', encoding=encoding)
        writer = csv.writer(f_out)
        writer.writerow(header)
        
        # Reset the progress bar (total_lines is approximate since header was removed)
        pbar = tqdm(total=total_lines, desc="Splitting CSV")
        
        # Iterate over each row in the input file
        for row in reader:
            writer.writerow(row)
            current_line += 1
            pbar.update(1)
            
            # When the current chunk reaches the specified number of lines, create a new file
            if current_line >= lines_per_file:
                f_out.close()
                file_count += 1
                current_line = 0
                output_file = os.path.join(output_folder, f"{output_prefix}_part{file_count}.csv")
                f_out = open(output_file, 'w', newline='', encoding=encoding)
                writer = csv.writer(f_out)
                writer.writerow(header)
        
        f_out.close()
        pbar.close()
    
    print(f"Finished splitting. Total files created: {file_count}")

# Define the file paths and parameters
input_csv = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\TRI_Preqin_Comprehensive_Matched.csv"
output_folder = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\Chunks"
output_prefix = "TRI_Preqin_chunk"
lines_per_chunk = 200000  # Adjust as needed

# Run the split function
split_csv(input_csv, output_folder, output_prefix, lines_per_file=lines_per_chunk)


# In[4]:


import csv
import os
from tqdm import tqdm

def split_csv(input_file, output_folder, output_prefix, lines_per_file=200000, encoding='utf-8'):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # First, count total lines (for progress bar accuracy)
    total_lines = 0
    with open(input_file, 'r', encoding=encoding) as f:
        for _ in f:
            total_lines += 1
    # Subtract one for the header
    total_lines -= 1
    print(f"Total data lines (excluding header): {total_lines}")
    
    with open(input_file, 'r', newline='', encoding=encoding) as f_in:
        reader = csv.reader(f_in)
        header = next(reader)  # store header
        
        file_count = 1
        line_count = 0
        output_file = os.path.join(output_folder, f"{output_prefix}_part{file_count}.csv")
        f_out = open(output_file, 'w', newline='', encoding=encoding)
        writer = csv.writer(f_out)
        writer.writerow(header)
        
        pbar = tqdm(total=total_lines, desc="Splitting CSV")
        for row in reader:
            writer.writerow(row)
            line_count += 1
            pbar.update(1)
            
            if line_count >= lines_per_file:
                f_out.close()
                print(f"Chunk {file_count} completed with {line_count} lines.")
                file_count += 1
                line_count = 0
                output_file = os.path.join(output_folder, f"{output_prefix}_part{file_count}.csv")
                f_out = open(output_file, 'w', newline='', encoding=encoding)
                writer = csv.writer(f_out)
                writer.writerow(header)
        
        f_out.close()
        pbar.close()
    print(f"Finished splitting. Total files created: {file_count}")

# Parameters
input_csv = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\TRI_Preqin_Comprehensive_Matched.csv"
output_folder = r"C:\Users\A-2020\OneDrive - King's College London\Desktop\Upgrade Data\Upgrade DATA\TRI\Chunks"
output_prefix = "TRI_Preqin_chunk"
lines_per_chunk = 200000  # You can adjust this value

split_csv(input_csv, output_folder, output_prefix, lines_per_file=lines_per_chunk)


# In[ ]:




