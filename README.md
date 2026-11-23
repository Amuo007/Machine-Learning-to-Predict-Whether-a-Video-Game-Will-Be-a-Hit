# Machine Learning to Predict Whether a Video Game Will Be a Hit

COSC 3337 course project. We use machine learning to predict whether a video game will be a **hit** (1 million+ copies sold worldwide) using only information known **before release**: genre, platform, publisher, release year, and two features we engineered.

## Questions
1. Which model predicts hits best?
2. Which features matter most?

## Dataset
- **Source:** [Video Game Sales (Kaggle)](https://www.kaggle.com/datasets/gregorut/videogamesales), scraped from VGChartz
- **Original size:** 16,598 games, 11 columns
- **After cleaning:** 15,849 games
- **Target:** `Hit` = 1 if `Global_Sales` ≥ 1 million copies, else 0 (12.9% of games are hits)

## Project Structure
```
├── data/
│   ├── vgsales.csv          # original dataset
│   ├── train.csv            # training data (80%)
│   └── test.csv             # test data (20%)
├── charts/                  # all saved charts
├── results/
│   ├── best_settings.json   # best hyperparameters per model
│   ├── model_scores.csv     # 5-fold scores for all models
│   └── ablation_scores.csv  # ablation results
├── 01_cleaning_and_charts.ipynb
├── 02_models.ipynb
└── 03_ablation.ipynb
```

## Notebooks
### 01: Data Cleaning, Feature Engineering, and EDA
- Removed rows with missing Year or Publisher, and "Unknown" publishers
- Removed one duplicate (same name, platform, and year)
- Corrected 4 wrong release years using online sources
- Removed 2016 games (less than a year on sale when the data was collected)
- Renamed a remake that shared its name with an older game
- Created the target `Hit`
- Engineered 2 features:
  - `Num_Platforms`: number of platforms the game is on so far (up to that year)
  - `Already_Released`: 1 if the game was already released on another platform in an earlier year
- Charts with written learnings (hit rate by genre, platform, publisher, year, and more)
- Split 80/20, keeping all versions of the same game on the same side
- Grouped publishers with fewer than 10 training games into "Other"

### 02: Model Training and Comparison
- Preprocessing: one-hot encoding for Platform, Genre, Publisher; standardization for Year and Num_Platforms
- 5-fold cross-validation (`StratifiedGroupKFold`), keeping game versions together
- Hyperparameter tuning with `GridSearchCV`, optimizing F1
- Models: Decision Tree, KNN, Logistic Regression, Random Forest, Neural Network
- Final evaluation of the best model on the held-out test set

### 03: Feature Ablation
- Retrained all 5 models with each feature removed, one at a time
- Measured the F1 drop for each feature

## Results
### 5-Fold Cross-Validation
| Model | Accuracy | Precision | Recall | F1 | Train F1 |
|---|---|---|---|---|---|
| **Random Forest** | 0.819 | 0.379 | 0.609 | **0.467** | 0.701 |
| Neural Network | 0.865 | 0.478 | 0.381 | 0.423 | 0.771 |
| Logistic Regression | 0.722 | 0.285 | 0.755 | 0.414 | 0.425 |
| Decision Tree | 0.736 | 0.280 | 0.660 | 0.393 | 0.566 |
| KNN | 0.818 | 0.339 | 0.424 | 0.376 | 0.632 |
| Baseline (always "not a hit") | 0.870 | 0.000 | 0.000 | 0.000 | 0.000 |

The baseline has the highest accuracy but never finds a hit, which is why we use F1 as our main metric.

### Final Test (Random Forest)
| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| 0.823 | 0.374 | 0.624 | **0.468** |

The test F1 matches the 5-fold F1, so the model generalizes well to unseen games. It beats the baseline and meets our goal of F1 ≥ 0.40.

### Feature Ablation (Random Forest, F1 drop when removed)
| Feature | F1 Drop |
|---|---|
| Publisher | 0.137 |
| Platform | 0.063 |
| Genre | 0.049 |
| Num_Platforms | 0.023 |
| Year | 0.017 |
| Already_Released | 0.002 |

**Publisher is the most important feature for all 5 models.** Already_Released adds almost nothing.

## Key Findings
- Hits are rare: only about 13% of games sell 1 million+ copies
- Nintendo stands out: about half of its games are hits
- Platform (platformer) and Shooter games have the highest hit rates; Adventure and Strategy the lowest
- Games released on more platforms are more likely to be hits
- Random Forest gives the best balance between finding hits and avoiding false alarms

## Limitations
- Sales are estimates from VGChartz, mostly physical copies, not digital
- The data ends in 2016 and doesn't cover newer consoles
- A hit is based on copies sold, not profit (no budget or price data)
- No data on reviews, marketing, or game quality
- Some games share a name with an older, different game, which can affect `Num_Platforms` and `Already_Released`
- Older games may look more successful because the website mostly records well-known older titles
- Most models show some overfitting (training F1 higher than quiz F1)

## How to Run
1. Install the libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Run the notebooks in order: `01` → `02` → `03`

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter

## Team
- Amrinder Singh
- Chuka Nwobu