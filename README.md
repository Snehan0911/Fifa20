# ⚽ FIFA 20 Players Data Analysis & Clustering Project

📌 Project Overview
This repository contains a comprehensive data analysis and machine learning clustering project on the FIFA 20 players dataset (`players_20.csv`). The goal is to analyze player attributes, discover trends, answer tactical/business questions, and group players into distinct playing styles (archetypes) using clustering.

All tasks have been consolidated into a single executable Python script [fifa20_analysis.py](file:///d:/Projects/Fifa20/fifa20_analysis.py), and are also detailed inside the Jupyter Notebook [PRCP-Fifa20.ipynb](file:///d:/Projects/Fifa20/PRCP-Fifa20.ipynb).

---

# 📂 Dataset Information

### FIFA 20 Football is arguably the most popular sport in the world and FIFA is the most popular football (soccer) simulation game by Electronic Arts (EA Sports).
### The dataset provided includes the players data for the Career Mode from FIFA 15 to FIFA 20 ("players_20.csv"). The data allows multiple comparisons of the same players across the last 6 versions of the videogame.
## Some ideas of possible analysis:
- Historical comparison between Messi and Ronaldo (what skill attributes changed the most during time - compared to real-life stats);
- Ideal budget to create a competitive team (at the level of top n teams in Europe) and at which point the budget does not allow to buy significantly better players for the 11-men lineup. An extra is the same comparison with the Potential attribute for the lineup instead of the Overall attribute;
- Sample analysis of top n% players (e.g. top 5% of the player) to see if some important attributes such as Agility or BallControl or Strength have been popular or not across the FIFA versions. An example would be seeing that the top 5% players of FIFA 20 are faster (higher Acceleration and Agility) compared to FIFA 15. The trend of attributes is also an important indication of how some attributes are necessary for players to win games (a version with more top 5% players with high BallControl stats would indicate that the game is more focused on the technique rather than the physical aspect).
### 🌍Domain: Sports
### Link : https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1004-Fifa20.zip


## attributes used in the project are: 
- Name: Name of the player. 
- Age: Age of the player.
- Height: Height of the player in inches (transformed to centimeters in preprocessing).
- Overall: General performance quality and value of the player representing the key positional skills and international reputation rated between 1-99. Overall attribute is used only in preprocessing and discussion stages because using it in modelling could lead to domination by this feature. The aim of the project is not basically sort and categorize the players using their overall talent and international reputation, but to cluster them based on using their whole skillset.
- Potential: Maximum Overall rating expected to be reached by a player in the top of his career rated between 1-99.
- PreferredFoot: Right or Left. Label encoder is applied as 0 for left and 1 for right.
- WeakFoot: Represents how well a player uses his weak foot (e.g. left for righties) rated between 1 to 5.
- WorkRate: Degree of the effort the player puts in terms of attack and defense rated as low, medium and high. This feature is divided into two new features as AttackWorkRate and DefenseWorkRate. Besides, label encoder is applied as 0 for low, 0.5 for medium and 1 for high.
- Position: Position of the players on the pitch which determines their roles and responsibilities in the team. Forward positions in the football and FIFA 19 can be grouped as striker (ST: center striker, RS: right striker, LS: left striker), forward (CF: center forward, RF: right forward, LF: left forward) and winger (RW: right winger, LW: left winger). The word, forward, is used both as a general term and a special position. Strikers are positioned in front of forwards and wingers and very closed to the opposing goal. Their main responsibilities are attacking and scoring goals, that’s why their ball control, shooting and finishing skills are expected to be well. Center forwards are positioned right behind the strikers. They are expected to receive balls from the others and score assists to the others or goals. In addition to the skills expected from strikers, they have to be good at passing. Right forwards and left forwards are positioned at the right and left of the center forwards with the same expectations. Wingers are positioned near the touchlines to create chances for strikers and forwards from the right and left side of the field by breakthrough and crosses and to score goals. They are expected to be good at dribbling, acceleration, passing and crossing. Positions are used only in preprocessing and discussion stages. 
- ST: Positional skill. Player’s general ability when playing in ST position rated between 1-99.
- RS: Positional skill. Player’s general ability when playing in in RS position rated between 1-99.
- LS: Positional skill. Player’s general ability when playing in in LS position rated between 1-99.
- CF: Positional skill. Player’s general ability when playing in in CF position rated between 1-99.
- RF: Positional skill. Player’s general ability when playing in in RF position rated between 1-99.
- LF: Positional skill. Player’s general ability when playing in in LF position rated between 1-99.
- RW: Positional skill. Player’s general ability when playing in in RW position rated between 1-99.
- LW: Positional skill. Player’s general ability when playing in in LW position rated between 1-99.
- Crossing: Crossing skill of the player rated between 1-99. Cross is a long-range pass from wings to center.
- Finishing: Finishing skill of the player rated between 1-99. Finishing in football refers to finish an attack by scoring a goal.
- HeadingAccuracy: Player’s accuracy to pass or shoot by using his head rated between 1-99.
- ShortPassing: Player’s accuracy for short passes rated between 1-99.
- LongPassing: Player’s accuracy for long passes rated between 1-99.
- Dribbling: Dribbling skill of the player rated between 1-99. Dribbling is carrying the ball without losing while moving in one particular direction.
- SprintSpeed: Speed rate of the player rated between 1-99.
- Acceleration: Shows how fast a player can reach his maximum sprint speed rated between 1-99.
- FKAccuracy: Player’s accuracy to score free kick goals rated between 1-99.
- BallControl: Player’s ability to control the ball rated between 1-99.
- Balance: Player’s ability to remain steady while running, carrying and controlling the ball rated between 1-99.
- ShotPower: Player’s strength level of shooting the ball rated between 1-99.
- Jumping: Player’s jumping skill rated between 1-99.
- Penalties: Player’s accuracy to score goals from penalty rated between 1-99.
- Strength: Physical strength of the player rated between 1-99.
- Agility: Gracefulness and quickness of the player while controlling the ball rated between 1-99.
- Reactions: Acting speed of the player to what happens in his environment rated between 1-99.
- Aggression: Aggression level of the player while pushing, pulling and tackling rated between 1-99.
- Positioning: Player’s ability to place himself in the right position to receive the ball or score goals rated between 1-99.
- Vision: Player’s mental awareness about the other players in the team for passing rated between 1-99.
- Volleys: Player’s ability to perform volleys rated between 1-99.
- LongShots: Player’s accuracy of shoots from long distances rated between 1-99.
- Stamina: Player’s ability to sustain his stamina level during the match rated between 1-99. Players with lower stamina get tired fast.
- Composure: Player’s ability to control his calmness and frustration during the match rated between 1-99.
- Curve: Player’s ability to curve the ball while passing or shooting rated between 1-99.
- Interceptions: Player’s ability to intercept the ball while opposite team’s players are passing rated between 1-99. It is a defensive skill.
- StandingTackle: Player’s ability to perform tackle (take the ball from the opposite player) while standing rated between 1-99. It is a defensive skill.
- SlidingTackle: Player’s ability to perform tackle by sliding rated between 1-99. It is a defensive skill.
- Marking: Player’s ability to apply strategies to prevent opposing team from taking the ball rated between 1-99. It is a defensive skill.

----

## 🎯 Project Objectives

## 📊 Task 1: Data Analysis
- Perform data cleaning and preprocessing.
- Handle missing values and duplicate records.
- Conduct Exploratory Data Analysis (EDA).
- Generate statistical summaries and visualizations.
  
## 🤖 Task 2: Player Clustering
- Select relevant player skill attributes.
- Apply feature scaling techniques.
- Implement clustering algorithms such as K-Means and Hierarchical Clustering.
- Evaluate cluster quality using metrics like Silhouette Score and Elbow Method.
- Group players based on their playing style and skill characteristics.

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
  
## ⚽ Task 3:Business Questions
- Identify the top 10 countries producing the highest number of professional football players.
- Analyze the relationship between player age and overall rating.
- Determine the age at which players typically stop improving.
- Compare salaries/wages among offensive positions such as Strikers (ST), Right Wingers (RW), and Left Wingers (LW).

#### 1. Top Player-Producing Countries
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

## 🛠️ Technologies Used

- 🐍 Python
- 📊 Pandas
- 🔢 NumPy
- 📈 Matplotlib
- 🎨 Seaborn
- 🤖 Scikit-Learn
- 📓 Jupyter Notebook
- 💻 VS Code
- 🔗 Git & GitHub

## 📈 Key Insights Generated

- 🌎 Country-wise Football Talent Distribution
- 📊 Age vs Overall Rating Analysis
- 💰 Position-wise Salary Comparison
- ⚽ Player Skill Analysis
- 🎯 Identification of Similar Player Groups
- 📉 Correlation Analysis of Football Attributes

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

## 💡 Challenges Faced

-🔹 Handling Missing Values

-🔹 High-Dimensional Data

-🔹 Feature Selection

-🔹 Determining Optimal Number of Clusters

-🔹 Outlier Detection and Treatment

# Author: Sneha Nuchha

# ⭐ If you found this project useful, don't forget to Star the Repository!
