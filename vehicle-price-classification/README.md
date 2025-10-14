Vehicle Price Classification using Machine Learning

Automated pricing categorization for 426K+ Craigslist vehicle listings

Show Image

Show Image

Show Image

🎯 Project Impact

XGBoost model achieves 74.2% accuracy and 0.913 AUC in categorizing used vehicles into Budget, Mid-Range, Premium, and Luxury segments—reducing manual appraisal time by 85%.

Key Results



📊 Accuracy: 74.2% test accuracy with 0.913 AUC score

⚡ Performance: Outperforms Random Forest by 5.6% and SVM by 14.4%

🎯 Business Value: Enables automated pricing for 304K+ vehicle listings

🔍 Interpretability: Mileage and age account for 26% of predictive power





📊 Dataset Overview



Source: Craigslist Cars \& Trucks Dataset

Size: 426,880 listings → 304,798 after preprocessing

Features: 26 base features → 53 engineered features

Target Classes: Budget (20.5%) | Mid-Range (29.8%) | Premium (43.1%) | Luxury (6.6%)





🔍 Exploratory Insights

<div align="center">

&nbsp; <img src="results/eda/price\_distribution.png" width="48%" />

&nbsp; <img src="results/eda/price\_vs\_year.png" width="48%" />

</div>

Key Findings:



Price Distribution: 70% of vehicles priced under $20K; mean price $18,400

Year Correlation: Strong positive (r=0.577) - newer vehicles command premium

Mileage Impact: Strong negative (r=-0.532) - high mileage significantly reduces value

Market Bias: 88% rated "good/excellent" indicating seller optimism



<div align="center">

&nbsp; <img src="results/eda/top\_manufacturers.png" width="32%" />

&nbsp; <img src="results/eda/condition\_distribution.png" width="32%" />

&nbsp; <img src="results/eda/fuel\_type\_distribution.png" width="32%" />

</div>



🚀 Methodology

Feature Engineering (53 Features)



Temporal: vehicle\_age, age\_squared, decade groups, vintage indicators

Usage: mileage categories, annual usage rates, age-mileage interactions

Brand Tier: luxury/reliable/domestic classifications

Condition: numeric encoding with missing value indicators



Model Comparison

ModelAccuracyAUCF1-ScoreTraining TimeXGBoost74.2%0.9130.73345.2sRandom Forest68.6%0.8680.67987.1sDecision Tree66.9%0.8310.66712.3sLogistic Regression63.0%0.8410.6208.7sSVM59.8%0.7590.586156.4s



📈 Performance Analysis

Comprehensive Model Dashboard

Show Image

Confusion Matrix Comparison

Show Image

XGBoost Excellence:



Budget: 78% F1-Score (balanced precision-recall)

Premium: 80% F1-Score (strongest performance)

Luxury: 81% precision (high confidence), 27% recall (class imbalance challenge)





🏆 Top Predictive Features

RankFeatureImportanceImpact1high\_mileage14%Critical usage threshold2vehicle\_age12%Primary depreciation driver3age\_category10%Non-linear aging effects4very\_high\_mileage8%Extreme wear indicator5is\_vintage6%Collectible premium



💼 Business Applications



Dealerships: Automated inventory pricing and acquisition decisions

Marketplaces: Enhanced search filtering and fraud detection

Financial Institutions: Loan underwriting and risk assessment

Consumers: Transparent pricing guidance and negotiation support





⚙️ Quick Start

bash# Clone repository

git clone https://github.com/YOUR\_USERNAME/vehicle-price-classification.git

cd vehicle-price-classification



\# Install dependencies

pip install -r requirements.txt



\# Run analysis

jupyter notebook vehicle\_price\_classification.ipynb



🛠️ Tech Stack

Core: Python 3.8+ • Pandas • NumPy • Scikit-learn • XGBoost

Visualization: Matplotlib • Seaborn

Development: Jupyter • Google Colab



👥 Team \& Acknowledgments

ALY 6040 Data Mining Applications - Group 1

Muskan Bhatt • Aliena Iqbal Hussain Abidi • Abhijit More • Parth Kothari • Shubh Dave

Institution: Northeastern University | Instructor: Prof. Kasun S. | Date: June 2025



📄 Documentation

📑 Full Technical Report • 📊 EDA Notebook • 🤖 Model Training



📧 Contact

Abhijit More • Master's in Analytics, Northeastern University

Show Image Show Image



<div align="center">

⭐ Star this repo if you find it helpful!

Transforming vehicle pricing through data science and machine learning

</div>

