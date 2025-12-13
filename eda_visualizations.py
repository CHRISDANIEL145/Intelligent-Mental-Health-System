# =====================================================================
# MENTAL HEALTH PROJECT - EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
# Generates charts for final-year project report
# =====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Use non-interactive backend for saving figures
import matplotlib
matplotlib.use('Agg')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =====================================================================
# LOAD ALL DATASETS
# =====================================================================
def load_datasets():
    """Load all 4 datasets"""
    datasets = {
        'students': pd.read_csv('dataset/students_sleep_screen.csv'),
        'social_media': pd.read_csv('dataset/social_media_depression.csv'),
        'burnout': pd.read_csv('dataset/burnout_medical_post_covid.csv'),
        'mbsr': pd.read_csv('dataset/mbsr_healthcare.csv')
    }
    print("✓ All datasets loaded successfully")
    for name, df in datasets.items():
        print(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns")
    return datasets

# =====================================================================
# 1. SCREEN TIME VS SLEEP QUALITY ANALYSIS
# =====================================================================
def analyze_screen_time_sleep(df):
    """
    Problem 1: Correlation Between Screen Time and Sleep Quality
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: SCREEN TIME VS SLEEP QUALITY (Students)")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1a. Correlation Heatmap
    corr_cols = ['screen_time_hours', 'screen_time_night_hours', 
                 'social_media_hours', 'sleep_duration_hours', 
                 'sleep_quality_score', 'insomnia_score', 'stress_score']
    corr_matrix = df[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                fmt='.2f', ax=axes[0, 0], square=True)
    axes[0, 0].set_title('Correlation Heatmap: Screen Time & Sleep Factors', fontsize=12)
    
    # 1b. Screen Time vs Sleep Quality Scatter
    axes[0, 1].scatter(df['screen_time_hours'], df['sleep_quality_score'], 
                       alpha=0.6, c=df['stress_score'], cmap='coolwarm')
    axes[0, 1].set_xlabel('Total Screen Time (hours)')
    axes[0, 1].set_ylabel('Sleep Quality Score')
    axes[0, 1].set_title('Screen Time vs Sleep Quality (color=stress)')
    
    # Add regression line
    z = np.polyfit(df['screen_time_hours'], df['sleep_quality_score'], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(df['screen_time_hours'].sort_values(), 
                    p(df['screen_time_hours'].sort_values()), 
                    "r--", alpha=0.8, label=f'Trend line')
    axes[0, 1].legend()
    
    # 1c. Night Screen Time Distribution by Sleep Label
    df['sleep_label'] = (df['sleep_quality_score'] >= 7.5).astype(int)
    df['sleep_category'] = df['sleep_label'].map({0: 'Poor Sleep', 1: 'Good Sleep'})
    
    sns.boxplot(data=df, x='sleep_category', y='screen_time_night_hours', ax=axes[1, 0])
    axes[1, 0].set_title('Night Screen Time by Sleep Quality')
    axes[1, 0].set_xlabel('Sleep Category')
    axes[1, 0].set_ylabel('Night Screen Time (hours)')
    
    # 1d. Sleep Quality Distribution
    sns.histplot(data=df, x='sleep_quality_score', hue='gender', kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Sleep Quality Distribution by Gender')
    axes[1, 1].axvline(x=7.5, color='red', linestyle='--', label='Threshold (7.5)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('fig_screen_time_sleep.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistical Analysis
    print("\n--- Statistical Summary ---")
    corr, p_val = stats.pearsonr(df['screen_time_hours'], df['sleep_quality_score'])
    print(f"Correlation (Screen Time vs Sleep Quality): r = {corr:.4f}, p = {p_val:.4f}")
    
    corr_night, p_night = stats.pearsonr(df['screen_time_night_hours'], df['sleep_quality_score'])
    print(f"Correlation (Night Screen vs Sleep Quality): r = {corr_night:.4f}, p = {p_night:.4f}")
    
    # T-test: Good vs Poor sleepers
    good_sleep = df[df['sleep_label'] == 1]['screen_time_night_hours']
    poor_sleep = df[df['sleep_label'] == 0]['screen_time_night_hours']
    t_stat, t_pval = stats.ttest_ind(good_sleep, poor_sleep)
    print(f"\nT-test (Night Screen Time: Good vs Poor Sleep):")
    print(f"  t-statistic = {t_stat:.4f}, p-value = {t_pval:.4f}")
    
    print("\n✓ Figure saved: fig_screen_time_sleep.png")
    return corr_matrix

# =====================================================================
# 2. MBSR EFFECTIVENESS ANALYSIS
# =====================================================================
def analyze_mbsr_effectiveness(df):
    """
    Problem 2: Effectiveness of MBSR in Healthcare Workers
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: MBSR EFFECTIVENESS (Healthcare Workers)")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 2a. Stress Score: MBSR vs Non-MBSR
    df['mbsr_group'] = df['mbsr_participation'].map({0: 'No MBSR', 1: 'MBSR Participant'})
    sns.boxplot(data=df, x='mbsr_group', y='stress_score', ax=axes[0, 0])
    axes[0, 0].set_title('Stress Score: MBSR vs Non-MBSR Groups')
    axes[0, 0].set_ylabel('Stress Score')
    
    # 2b. Burnout Score: MBSR vs Non-MBSR
    sns.boxplot(data=df, x='mbsr_group', y='burnout_score', ax=axes[0, 1])
    axes[0, 1].set_title('Burnout Score: MBSR vs Non-MBSR Groups')
    axes[0, 1].set_ylabel('Burnout Score')
    
    # 2c. MBSR Weeks vs Burnout (for participants only)
    mbsr_participants = df[df['mbsr_participation'] == 1]
    axes[1, 0].scatter(mbsr_participants['mbsr_weeks_completed'], 
                       mbsr_participants['burnout_score'], alpha=0.6)
    axes[1, 0].set_xlabel('MBSR Weeks Completed')
    axes[1, 0].set_ylabel('Burnout Score')
    axes[1, 0].set_title('MBSR Duration vs Burnout Score')
    
    # Add trend line
    if len(mbsr_participants) > 2:
        z = np.polyfit(mbsr_participants['mbsr_weeks_completed'], 
                       mbsr_participants['burnout_score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(mbsr_participants['mbsr_weeks_completed'].min(),
                            mbsr_participants['mbsr_weeks_completed'].max(), 100)
        axes[1, 0].plot(x_line, p(x_line), "r--", label='Trend')
        axes[1, 0].legend()
    
    # 2d. Pre vs Post COVID Burnout
    df['covid_period'] = df['post_covid_flag'].map({0: 'Pre-COVID', 1: 'Post-COVID'})
    sns.barplot(data=df, x='covid_period', y='burnout_score', hue='mbsr_group', ax=axes[1, 1])
    axes[1, 1].set_title('Burnout: Pre vs Post COVID (by MBSR Status)')
    axes[1, 1].set_ylabel('Mean Burnout Score')
    
    plt.tight_layout()
    plt.savefig('fig_mbsr_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistical Tests
    print("\n--- Statistical Analysis ---")
    mbsr_yes = df[df['mbsr_participation'] == 1]['burnout_score']
    mbsr_no = df[df['mbsr_participation'] == 0]['burnout_score']
    
    t_stat, t_pval = stats.ttest_ind(mbsr_yes, mbsr_no)
    print(f"T-test (Burnout: MBSR vs Non-MBSR):")
    print(f"  Mean MBSR: {mbsr_yes.mean():.2f}, Mean Non-MBSR: {mbsr_no.mean():.2f}")
    print(f"  t-statistic = {t_stat:.4f}, p-value = {t_pval:.4f}")
    
    if t_pval < 0.05:
        print("  → Statistically significant difference (p < 0.05)")
    else:
        print("  → No statistically significant difference (p >= 0.05)")
    
    print("\n✓ Figure saved: fig_mbsr_effectiveness.png")

# =====================================================================
# 3. SOCIAL MEDIA & DEPRESSION/ANXIETY ANALYSIS
# =====================================================================
def analyze_social_media_mental_health(df):
    """
    Problem 3 & 4: Social Media Impact on Depression/Anxiety
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: SOCIAL MEDIA & DEPRESSION/ANXIETY")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 3a. Social Media Hours vs PHQ-9 Score
    axes[0, 0].scatter(df['social_media_hours'], df['phq9_score'], 
                       alpha=0.6, c=df['depression_label'], cmap='RdYlGn_r')
    axes[0, 0].set_xlabel('Social Media Hours (daily)')
    axes[0, 0].set_ylabel('PHQ-9 Score (Depression)')
    axes[0, 0].set_title('Social Media Usage vs Depression Score')
    
    # 3b. Depression by Platform
    platform_depression = df.groupby('social_media_type_dominant')['depression_label'].mean() * 100
    platform_depression.sort_values(ascending=True).plot(kind='barh', ax=axes[0, 1], color='coral')
    axes[0, 1].set_xlabel('Depression Rate (%)')
    axes[0, 1].set_title('Depression Rate by Social Media Platform')
    
    # 3c. PHQ-9 and GAD-7 Distribution
    axes[1, 0].hist(df['phq9_score'], bins=20, alpha=0.7, label='PHQ-9 (Depression)', color='blue')
    axes[1, 0].hist(df['gad7_score'], bins=20, alpha=0.7, label='GAD-7 (Anxiety)', color='orange')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of PHQ-9 and GAD-7 Scores')
    axes[1, 0].legend()
    
    # 3d. Depression & Anxiety by Role
    role_stats = df.groupby('role')[['depression_label', 'anxiety_label']].mean() * 100
    role_stats.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Role')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_title('Depression & Anxiety Rates by Role')
    axes[1, 1].legend(['Depression', 'Anxiety'])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('fig_social_media_mental_health.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistics
    print("\n--- Statistical Summary ---")
    corr_dep, p_dep = stats.pearsonr(df['social_media_hours'], df['phq9_score'])
    print(f"Correlation (Social Media Hours vs PHQ-9): r = {corr_dep:.4f}, p = {p_dep:.4f}")
    
    corr_anx, p_anx = stats.pearsonr(df['social_media_hours'], df['gad7_score'])
    print(f"Correlation (Social Media Hours vs GAD-7): r = {corr_anx:.4f}, p = {p_anx:.4f}")
    
    print(f"\nDepression Rate: {df['depression_label'].mean()*100:.1f}%")
    print(f"Anxiety Rate: {df['anxiety_label'].mean()*100:.1f}%")
    
    print("\n✓ Figure saved: fig_social_media_mental_health.png")


# =====================================================================
# 4. BURNOUT POST-COVID ANALYSIS
# =====================================================================
def analyze_burnout_post_covid(df):
    """
    Problem 5: Burnout Among Medical Professionals Post-COVID
    """
    print("\n" + "="*70)
    print("ANALYSIS 4: BURNOUT POST-COVID (Medical Professionals)")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 4a. Burnout Distribution
    df['burnout_category'] = df['burnout_label'].map({0: 'Low Burnout', 1: 'High Burnout'})
    burnout_counts = df['burnout_category'].value_counts()
    axes[0, 0].pie(burnout_counts, labels=burnout_counts.index, autopct='%1.1f%%',
                   colors=['lightgreen', 'salmon'], explode=[0, 0.05])
    axes[0, 0].set_title('Burnout Distribution (Post-COVID)')
    
    # 4b. Work Hours vs Burnout Score
    axes[0, 1].scatter(df['work_hours_per_week'], df['burnout_score'],
                       alpha=0.6, c=df['burnout_label'], cmap='RdYlGn_r')
    axes[0, 1].set_xlabel('Work Hours per Week')
    axes[0, 1].set_ylabel('Burnout Score')
    axes[0, 1].set_title('Work Hours vs Burnout Score')
    axes[0, 1].axhline(y=60, color='red', linestyle='--', alpha=0.5, label='High Burnout Threshold')
    axes[0, 1].legend()
    
    # 4c. Burnout by Role
    role_burnout = df.groupby('role')['burnout_score'].agg(['mean', 'std'])
    role_burnout['mean'].plot(kind='bar', yerr=role_burnout['std'], ax=axes[1, 0],
                               capsize=5, color=['steelblue', 'coral'])
    axes[1, 0].set_xlabel('Role')
    axes[1, 0].set_ylabel('Mean Burnout Score')
    axes[1, 0].set_title('Burnout Score by Role')
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # 4d. Factors Contributing to Burnout (Correlation)
    factors = ['stress_score', 'work_hours_per_week', 'patient_load_per_week', 'mbsr_participation']
    correlations = df[factors + ['burnout_score']].corr()['burnout_score'][:-1]
    correlations.sort_values().plot(kind='barh', ax=axes[1, 1], color='teal')
    axes[1, 1].set_xlabel('Correlation with Burnout Score')
    axes[1, 1].set_title('Factors Contributing to Burnout')
    axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('fig_burnout_post_covid.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistics
    print("\n--- Statistical Summary ---")
    print(f"Total Medical Professionals: {len(df)}")
    print(f"High Burnout Rate: {df['burnout_label'].mean()*100:.1f}%")
    print(f"Mean Burnout Score: {df['burnout_score'].mean():.2f} (SD: {df['burnout_score'].std():.2f})")
    print(f"Mean Work Hours: {df['work_hours_per_week'].mean():.1f} hours/week")
    print(f"Mean Patient Load: {df['patient_load_per_week'].mean():.1f} patients/week")
    
    # MBSR effect
    mbsr_yes = df[df['mbsr_participation'] == 1]['burnout_score']
    mbsr_no = df[df['mbsr_participation'] == 0]['burnout_score']
    t_stat, t_pval = stats.ttest_ind(mbsr_yes, mbsr_no)
    print(f"\nMBSR Effect on Burnout:")
    print(f"  With MBSR: {mbsr_yes.mean():.2f}, Without MBSR: {mbsr_no.mean():.2f}")
    print(f"  t-statistic = {t_stat:.4f}, p-value = {t_pval:.4f}")
    
    print("\n✓ Figure saved: fig_burnout_post_covid.png")

# =====================================================================
# 5. COMPREHENSIVE SUMMARY STATISTICS
# =====================================================================
def generate_summary_statistics(datasets):
    """Generate summary statistics for all datasets"""
    print("\n" + "="*70)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("="*70)
    
    for name, df in datasets.items():
        print(f"\n--- {name.upper()} DATASET ---")
        print(f"Shape: {df.shape}")
        print(f"\nNumerical Summary:")
        print(df.describe().round(2).to_string())
        print("\n" + "-"*50)

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def run_all_analyses():
    """Run complete EDA for all problem statements"""
    
    print("\n" + "="*70)
    print("MENTAL HEALTH PROJECT - COMPLETE EDA")
    print("="*70)
    
    # Load data
    datasets = load_datasets()
    
    # Run analyses
    analyze_screen_time_sleep(datasets['students'])
    analyze_mbsr_effectiveness(datasets['mbsr'])
    analyze_social_media_mental_health(datasets['social_media'])
    analyze_burnout_post_covid(datasets['burnout'])
    
    # Summary
    generate_summary_statistics(datasets)
    
    print("\n" + "="*70)
    print("✅ ALL ANALYSES COMPLETE")
    print("Generated Figures:")
    print("  1. fig_screen_time_sleep.png")
    print("  2. fig_mbsr_effectiveness.png")
    print("  3. fig_social_media_mental_health.png")
    print("  4. fig_burnout_post_covid.png")
    print("="*70)

if __name__ == "__main__":
    run_all_analyses()
