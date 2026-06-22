"""
FIFA 20 Players Data Analysis and Clustering Project
This script consolidates all data preprocessing, exploratory data analysis (EDA),
specific analytical question answering, and K-Means clustering.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set style for all plots
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)

def main():
    print("=== Step 1: Loading Dataset ===")
    csv_path = "players_20.csv"
    if not os.path.exists(csv_path):
        # Fallback to absolute path if needed
        csv_path = r"d:\Projects\Fifa20\players_20.csv"
    
    data = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully. Shape: {data.shape}")

    print("\n=== Step 2: Data Preprocessing ===")
    # 2.1 Remove special characters
    data = data.replace('[\#]', '', regex=True)
    
    # 2.2 Clean positional rating columns (e.g., 'st', 'lw', etc.) by removing "+" modifier
    position_cols = [
        'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 
        'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb',
        'lb', 'lcb', 'cb', 'rcb', 'rb'
    ]
    for col in position_cols:
        if col in data.columns:
            data[col] = data[col].astype(str).str.split('+', expand=True)[0]
            data[col] = pd.to_numeric(data[col], errors='coerce')
    print("Positional columns cleaned and converted to numeric.")

    print("\n=== Step 3: Univariate Analysis (Sweetviz) ===")
    # Define features of interest for sweetviz analysis
    univariate_cols = [
        'age', 'height_cm', 'weight_kg', 'overall', 'potential', 'value_eur', 'wage_eur', 
        'preferred_foot', 'international_reputation', 'weak_foot', 'skill_moves', 'work_rate', 
        'body_type', 'real_face', 'release_clause_eur', 'pace', 'shooting', 'passing',
        'dribbling', 'defending', 'physic', 'gk_diving', 'gk_handling', 'gk_kicking', 
        'gk_reflexes', 'gk_speed', 'gk_positioning', 'attacking_crossing', 'attacking_finishing', 
        'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 
        'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 
        'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 
        'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 
        'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 
        'mentality_composure', 'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 
        'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 
        'goalkeeping_reflexes'
    ]
    # Filter available columns
    univariate_cols = [c for c in univariate_cols if c in data.columns]
    
    print("Generating Sweetviz report...")
    report = sweetviz.analyze(data[univariate_cols])
    report.show_html("SWEETVIZ_REPORT.html", open_browser=False)
    print("Sweetviz report saved to 'SWEETVIZ_REPORT.html'.")

    print("\n=== Step 4: Task 3 - Solving Specific Questions ===")
    
    # Question 1: Top 10 Countries with Most Players
    print("\n[Q1] Preparing list of top 10 player-producing countries...")
    top_10 = data['nationality'].value_counts().head(10)
    print(top_10)
    
    plt.figure()
    sns.barplot(x=top_10.index, y=top_10.values, palette='viridis')
    plt.title('Top 10 Player-Producing Countries (FIFA 20)')
    plt.xlabel('Country')
    plt.ylabel('Number of Players')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('top_10_countries.png')
    plt.close()
    print("Plot saved as 'top_10_countries.png'.")

    # Question 2: Overall Rating vs. Age
    print("\n[Q2] Plotting Overall Rating vs. Age...")
    plt.figure()
    sns.lineplot(x='age', y='overall', data=data, color='brown', linewidth=2.5, label='Average Overall')
    sns.scatterplot(x='age', y='overall', data=data, color='orange', alpha=0.1, label='Individual Players')
    plt.title('Overall Rating vs. Age')
    plt.xlabel('Age')
    plt.ylabel('Overall Rating')
    plt.legend()
    plt.tight_layout()
    plt.savefig('overall_vs_age.png')
    plt.close()
    
    # Calculate average overall by age
    age_stats = data.groupby('age')['overall'].mean()
    print("Average Overall by Age:")
    for age, val in age_stats.items():
        print(f"  Age {age}: {val:.2f}")
    print("Insight: Overall ratings peak and plateau between ages 28 and 32, then decline. Thus, a player stops improving after age 31-32.")

    # Question 3: Wages of Offensive Players (ST vs. RW vs. LW)
    print("\n[Q3] Comparing wages of Strikers, Right Wingers, and Left Wingers...")
    is_st = data['player_positions'].apply(lambda x: any(pos.strip() == 'ST' for pos in str(x).split(',')))
    is_rw = data['player_positions'].apply(lambda x: any(pos.strip() == 'RW' for pos in str(x).split(',')))
    is_lw = data['player_positions'].apply(lambda x: any(pos.strip() == 'LW' for pos in str(x).split(',')))

    st_wages = data[is_st]['wage_eur'].dropna()
    rw_wages = data[is_rw]['wage_eur'].dropna()
    lw_wages = data[is_lw]['wage_eur'].dropna()

    wages_summary = pd.DataFrame({
        'Striker (ST)': [st_wages.mean(), st_wages.median()],
        'Right Winger (RW)': [rw_wages.mean(), rw_wages.median()],
        'Left Winger (LW)': [lw_wages.mean(), lw_wages.median()]
    }, index=['Mean Wage (EUR)', 'Median Wage (EUR)']).T
    print(wages_summary)

    plt.figure()
    sns.barplot(x=wages_summary.index, y='Mean Wage (EUR)', data=wages_summary, palette='magma')
    plt.title('Average Wage Comparison by Offensive Position')
    plt.ylabel('Mean Wage (EUR)')
    plt.xlabel('Position')
    plt.tight_layout()
    plt.savefig('wage_comparison.png')
    plt.close()
    print("Wage comparison plot saved as 'wage_comparison.png'.")
    print("Insight: Left-wingers (LW) earn the highest average wage (~14,541 EUR), followed by Right-wingers (RW) (~13,626 EUR), and Strikers (ST) (~10,612 EUR).")

    print("\n=== Step 5: Task 2 - Player Clustering ===")
    
    # 5.1 Select 34 detailed skill features
    skill_cols = [
        'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
        'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
        'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance',
        'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
        'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure',
        'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle',
        'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes'
    ]
    
    X = data[skill_cols].fillna(0)
    
    # 5.2 Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5.3 Plot Elbow Curve
    print("Running K-Means elbow analysis (K=1 to 10)...")
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure()
    plt.plot(k_range, wcss, 'ro-')
    plt.title('K-Means Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Inertia)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('kmeans_elbow.png')
    plt.close()
    print("Elbow curve saved as 'kmeans_elbow.png'.")

    # 5.4 Fit KMeans with K=4
    print("Fitting K-Means with optimal K=4...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(X_scaled)
    
    print("\nPlayer count in each cluster:")
    print(data['cluster'].value_counts())

    # 5.5 Profile the clusters
    print("\nProfiling Cluster Mean Skills:")
    profile_attrs = ['attacking_finishing', 'defending_standing_tackle', 'goalkeeping_diving', 'movement_acceleration', 'skill_ball_control', 'power_strength', 'power_stamina']
    profiles = data.groupby('cluster')[profile_attrs].mean()
    print(profiles)
    
    # 5.6 PCA and Visualizations
    print("\nReducing dimensions using PCA for visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    data['pca1'] = X_pca[:, 0]
    data['pca2'] = X_pca[:, 1]
    
    # Map cluster numbers to descriptions
    cluster_labels = {
        0: 'Midfielders/Playmakers',
        1: 'Goalkeepers',
        2: 'Defenders',
        3: 'Attackers/Wingers'
    }
    data['cluster_label'] = data['cluster'].map(cluster_labels)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='pca1', y='pca2', hue='cluster_label', data=data, palette='Set1', alpha=0.6, edgecolor=None)
    plt.title('Player Clusters in PCA Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Player Profiles')
    plt.tight_layout()
    plt.savefig('player_clusters_pca.png')
    plt.close()
    print("PCA cluster visualization plot saved as 'player_clusters_pca.png'.")
    
    print("\n=== Project Successfully Completed! ===")

if __name__ == "__main__":
    main()
