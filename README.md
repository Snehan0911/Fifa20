# FIFA 20 Players Data Analysis & Clustering

This repository contains a comprehensive data analysis and machine learning clustering project on the FIFA 20 players dataset (`players_20.csv`). The goal is to analyze player attributes, discover trends, answer tactical/business questions, and group players into distinct playing styles (archetypes) using clustering.

All tasks have been consolidated into a single executable Python script [fifa20_analysis.py](file:///d:/Projects/Fifa20/fifa20_analysis.py), and are also detailed inside the Jupyter Notebook [PRCP-Fifa20.ipynb](file:///d:/Projects/Fifa20/PRCP-Fifa20.ipynb).

---

## 📊 Task 3: Key Insights & Questions Answered

### 1. Top Player-Producing Countries
A rank-ordered list of the top 10 countries producing the most players at the top level:
1. **England** - 1,667 players
2. **Germany** - 1,216 players
3. **Spain** - 1,035 players
4. **France** - 984 players
5. **Argentina** - 886 players
6. **Brazil** - 824 players
7. **Italy** - 732 players
8. **Colombia** - 591 players
9. **Japan** - 453 players
10. **Netherlands** - 416 players

*Insight: European and South American nations dominate player production, reflecting their deep-rooted footballing infrastructure, academies, and professional leagues.*

### 2. Overall Rating vs. Age (Player Development & Decline)
* The overall rating of players rises steeply from age 16 up to 25 as they develop.
* Player ratings peak and **plateau between the ages of 28 and 32** (maintaining an average overall rating of around 69.4 - 69.9).
* After **age 32**, the average overall rating begins to decline as physical attributes (such as acceleration, sprint speed, and stamina) deteriorate.
* **Conclusion: Age 31-32 is the age after which a player generally stops improving.**

### 3. Offensive Player Salaries (Striker vs. Wingers)
Comparing the average wages (EUR) of offensive player positions:
* **Left Winger (LW)**: Mean wage of **~14,541 EUR** (Median: 4,000 EUR)
* **Right Winger (RW)**: Mean wage of **~13,626 EUR** (Median: 4,000 EUR)
* **Striker (ST)**: Mean wage of **~10,612 EUR** (Median: 4,000 EUR)

*Insight: Modern football tactics heavily prioritize explosive, creative wingers who can break down compact defenses and cross/cut-in. This high demand and low supply makes wingers command higher average salaries than traditional center-forward strikers.*

---

## 🤖 Task 2: Player Clustering (Machine Learning)

To group players based on their style of play rather than just their overall rating or reputation, we conducted clustering on **34 detailed skill features** (e.g., crossing, finishing, dribbling, ball control, strength, tackling, and goalkeeping skills) using **K-Means Clustering**.

### 1. Determining Optimal K
Using the Elbow Method and Silhouette Analysis:
* **K=2** gives the highest Silhouette score (~0.55) because it separates Goalkeepers from Outfield players.
* **K=4** was chosen as the optimal tactical clustering. It has a high cohesion/separation and maps perfectly to the four fundamental football positions:

### 2. Cluster Profiles (Tactical Playing Styles)
* **Cluster 0: Midfielders & Playmakers (Box-to-Box)**
  * *Characteristics*: Extremely well-rounded. High stamina (74.0), short passing (70.7), and ball control (70.9). They contribute both offensively (finishing: 57.2) and defensively (tackling: 60.8).
  * *Primary Positions*: `CM`, `CDM`, `CAM`, `LM`, `RM`.
* **Cluster 1: Goalkeepers**
  * *Characteristics*: High goalkeeping diving (65.4), handling (63.1), kicking (61.8), positioning (63.4), and reflexes (66.4). Very low outfield values.
  * *Primary Positions*: `GK`.
* **Cluster 2: Defensive Specialists**
  * *Characteristics*: High physical strength (69.9), jumping (67.7), aggression (63.5), and standing tackles (64.0). Very low offensive stats (finishing: 31.7).
  * *Primary Positions*: `CB`, `RB`, `LB`.
* **Cluster 3: Attackers & Wingers**
  * *Characteristics*: High acceleration (71.9), sprint speed (71.7), agility (70.4), and finishing (59.8). Low defensive work and tackles (30.3).
  * *Primary Positions*: `ST`, `LM`, `RM`, `CAM`, `RW`, `LW`.

### 3. Dimensionality Reduction & Visualization
Using **PCA (Principal Component Analysis)**, the 34 dimensions were reduced to 2 components. The resulting scatter plot shows clear, distinct boundaries between the 4 playing styles, confirming that K-Means effectively grouped the players.

---

## ⚙️ How to Setup & Run the Project

### Prerequisites
* Anaconda or Python 3.10+
* Install dependencies listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### Activating Virtual Environment (Windows PowerShell)
If your script execution policy is restricted, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv\Scripts\Activate.ps1
```

### Running the Project Script
To run the full pipeline (preprocessing, EDA, visualization, Task 2 clustering, and Task 3 questions), run:
```bash
python fifa20_analysis.py
```
This generates and saves the Sweetviz report (`SWEETVIZ_REPORT.html`) and the following plots:
* `top_10_countries.png`
* `overall_vs_age.png`
* `wage_comparison.png`
* `kmeans_elbow.png`
* `player_clusters_pca.png`

### Running the Notebook
You can also run the Jupyter notebook `PRCP-Fifa20.ipynb` using JupyterLab or Jupyter Notebook:
```bash
jupyter notebook PRCP-Fifa20.ipynb
```
All code cells, EDA plots, Sweetviz reports, Elbow/Silhouette curves, and PCA cluster scatter plots are pre-computed and saved inside the notebook.
