import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gzip
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects.pandas2ri as pandas2ri
from google import genai
from google.genai.types import GenerateContentConfig

# Initialize Gemini Client

GEMINI_API_KEY = "AIzaSyCsu6vOCITGazfXwh3v6kKt0fbRmk2HoAw"
client = genai.Client(api_key=GEMINI_API_KEY)
config = GenerateContentConfig(
    system_instruction="You are an AI assistant that helps with genetic data analysis. Based on the analysis results, predict which SNPs are more likely to be causal.",
    tools=[]  # Here we don't need specific tools, Gemini will handle the prediction
)

# Activate R-Python converters
numpy2ri.activate()
pandas2ri.activate()

def load_gwas_data(file_path):
    """Load GWAS data from a compressed TSV file."""
    with gzip.open(file_path, 'rt') as f:
        gwas_data_df = pd.read_csv(f, sep='\t')
    return gwas_data_df

def preprocess_gwas_data(gwas_data_df):
    """Preprocess GWAS data by splitting variant info and renaming columns."""
    gwas_data_df['CHR'] = gwas_data_df['variant'].str.split(':').str[0]
    gwas_data_df['POS'] = gwas_data_df['variant'].str.split(':').str[1]
    gwas_data_df['A2'] = gwas_data_df['variant'].str.split(':').str[2]
    gwas_data_df['A1'] = gwas_data_df['variant'].str.split(':').str[3]
    gwas_data_df = gwas_data_df.rename(columns={'variant': 'SNPID', 'pval': 'P'})
    return gwas_data_df

def filter_significant_snps(gwas_data_df, maf_threshold=0.05, p_threshold=5e-8):
    """Filter significant SNPs based on MAF and p-value thresholds."""
    minor_af_filtered_df = gwas_data_df[gwas_data_df['minor_AF'] > maf_threshold]
    significant_snp_df = minor_af_filtered_df[minor_af_filtered_df['P'] <= p_threshold]
    significant_snp_df = significant_snp_df[~significant_snp_df['SNPID'].str.startswith('X:')]
    return significant_snp_df

def extract_region_snps(significant_snp_df, variant_position, window_size=500000):
    """Extract SNPs within a window around a variant position."""
    start_pos = variant_position - window_size
    end_pos = variant_position + window_size
    region_snp_df = significant_snp_df[
        (significant_snp_df['POS'] >= start_pos) & 
        (significant_snp_df['POS'] <= end_pos)
    ]
    region_snp_df["log_pvalue"] = -np.log10(region_snp_df["P"])
    return region_snp_df

def run_susie_analysis(snp_df, ld_matrix, n=503, L=10):
    """Run SuSiE analysis on SNP data with LD matrix."""
    susieR = importr('susieR')
    ro.r('set.seed(123)')
    fit = susieR.susie_rss(
        bhat=snp_df["beta"].values.reshape(len(snp_df), 1),
        shat=snp_df["se"].values.reshape(len(snp_df), 1),
        R=ld_matrix,
        L=L,
        n=n
    )
    return fit

def get_credible_sets(fit, ld_matrix, coverage=0.95, min_abs_corr=0.5):
    """Get credible sets from SuSiE fit."""
    susieR = importr('susieR')
    credible_sets = susieR.susie_get_cs(
        fit, 
        coverage=coverage, 
        min_abs_corr=min_abs_corr, 
        Xcorr=ld_matrix
    )
    return credible_sets

def predict_causal_variants(susie_results, region_snp_df):
    """Use Gemini to predict causal variants based on SuSiE results."""
    snp_info = region_snp_df[['SNPID', 'P', 'beta', 'se']].to_dict(orient='records')
    input_data = {
        'snp_info': snp_info,
        'susie_results': susie_results
    }
    
    # Generate causal prediction using Gemini
    r = client.models.generate_content(model="gemini-2.0-flash", config=config, contents=input_data)
    return r.text

def plot_susie_results(snp_df, fit, ld_matrix, col_to_plot="MLOG10P", window_size=500000):
    """Plot SuSiE results with credible sets."""
    if col_to_plot == "MLOG10P":
        snp_df[col_to_plot] = -np.log10(snp_df["P"])
    
    lead_pos = snp_df["P"].idxmin()
    lead_x = snp_df.loc[lead_pos, "POS"]
    lead_y = snp_df.loc[lead_pos, col_to_plot]
    
    credible_sets = get_credible_sets(fit, ld_matrix)[0]
    n_cs = len(credible_sets)
    
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(15, 7))
    
    p = ax.scatter(
        snp_df["POS"], 
        snp_df[col_to_plot], 
        c=ld_matrix[lead_pos]
    )
    
    ax.annotate(
        f"Lead Variant: {snp_df.loc[lead_pos, 'SNPID']}", 
        (lead_x, lead_y), 
        textcoords="offset points", 
        xytext=(0, lead_y + (0.02 if col_to_plot == "pip" else 2)), 
        ha='center', 
        fontsize=12
    )
    
    for i in range(n_cs):
        cs_index = credible_sets[i]
        pos = snp_df.loc[np.array(cs_index) - 1, "POS"]
        y = snp_df.loc[np.array(cs_index) - 1, col_to_plot]
        ax.scatter(
            pos, y, 
            marker='o', s=40, 
            label=f"CS{i+1}", 
            edgecolors="green", 
            facecolors="none"
        )
    
    plt.colorbar(p, label="LD to lead variant")
    ax.set_xlabel("Position")
    ax.set_ylabel(col_to_plot)
    ax.set_xlim((lead_x - window_size, lead_x + window_size))
    plt.legend()
    plt.show()

def main():
    # Load and preprocess GWAS data
    file_path = "../data/susie/gwas/21001_raw.gwas.imputed_v3.both_sexes.tsv.bgz"
    gwas_data_df = load_gwas_data(file_path)
    gwas_data_df = preprocess_gwas_data(gwas_data_df)
    
    # Filter significant SNPs
    significant_snp_df = filter_significant_snps(gwas_data_df)
    
    # Extract SNPs in a specific region (e.g., for a variant of interest)
    variant_position = 53828066
    region_snp_df = extract_region_snps(significant_snp_df, variant_position)
    
    # Run SuSiE analysis
    ld_r = np.random.rand(100, 100)  # Placeholder for LD matrix
    fit = run_susie_analysis(region_snp_df, ld_r)
    
    # Get credible sets
    credible_sets = get_credible_sets(fit, ld_r)
    
    # Predict causal variants using Gemini
    susie_results = " ".join([f"Credible set {i+1}: {cs}" for i, cs in enumerate(credible_sets)])
    causal_prediction = predict_causal_variants(susie_results, region_snp_df)
    
    # Display the results in Streamlit
    st.title("Causal Variant Prediction with SuSiE and Gemini")
    st.subheader("SuSiE Credible Sets")
    st.text(susie_results)
    
    st.subheader("Gemini Causal Prediction")
    st.text(causal_prediction)
    
    # Plot results
    plot_susie_results(region_snp_df, fit, ld_r, col_to_plot="MLOG10P")

if __name__ == "__main__":
    main()