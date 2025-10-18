Customer Churn Prediction using Time-Aware Machine Learning
Author: Yordanos Yhdego
Institution: [Add Institution Name]
Date: [Add Date]
 
Abstract
This report presents a time-aware machine learning approach to customer churn prediction for a utility company. Unlike conventional churn models that rely on static aggregates, this work enforces chronological splits to eliminate leakage and preserve temporal patterns. Two calibrated models—Random Forest and XGBoost with SMOTE—were compared using metrics such as PR-AUC, ROC-AUC, Brier score, and F1-macro. XGBoost+SMOTE-Cal achieved the best performance (PR-AUC ≈ 13.6%). The findings establish a robust, leak-safe baseline for future iterations with time-sensitive rolling features, expected to further improve predictive power.
1. Introduction
1.1 Problem Overview
Customer churn prediction plays a critical role in helping utility companies anticipate customer loss and allocate retention resources effectively. By identifying customers most likely to churn, companies can implement targeted retention campaigns to minimize revenue loss.
1.2 Limitations of Prior Work
Most churn studies collapse temporal data into static aggregates, discarding trends and temporal dependencies. Additionally, random train-test splits often cause target leakage, leading to overly optimistic results.
1.3 Project Objective
The goal of this project is to rebuild the churn prediction pipeline in a time-aware and leak-free manner. This iteration focuses on enforcing chronological splits, applying SMOTE only to the training set, and using calibrated probability outputs for business decision-making.
[Insert Visual 1: Comparison between Conventional and Time-Aware Churn Modeling]
2. Dataset
2.1 Data Description
The dataset consists of monthly customer reference snapshots (clients_ref) and corresponding price and usage data. Each record represents a customer's behavior up to a given reference month (m_ref).
2.2 Key Issues
The dataset contained missing values, zero-variance columns, and an imbalanced target variable where churn cases represent a small fraction of total observations. Additionally, prior versions suffered from aggregation leakage.
2.3 Preprocessing Summary
Preprocessing steps included imputing missing values, dropping zero-variance features, and applying SMOTE on the training subset to balance classes. Chronological 60/20/20 splits were used for train, validation, and test sets.
[Insert Visual 2: Class Distribution Bar Chart]
3. Methodology and System Design
3.1 Architecture Overview
The pipeline comprises data preparation, chronological splitting, SMOTE resampling, model training, calibration, and evaluation. Calibration was performed using isotonic regression on validation data to improve probability accuracy.
[Insert Visual 3: End-to-End Pipeline Diagram]
3.2 Model Training
Two models were trained and calibrated: Random Forest (RF+SMOTE-Cal) and XGBoost (XGB+SMOTE-Cal). Both models were evaluated using PR-AUC, ROC-AUC, Brier score, F1-macro, and Recall@Top-10%.
[Insert Visual 4: Chronological Split Illustration]
4. Results and Discussion
The results indicate that the XGB+SMOTE-Cal model outperformed RF+SMOTE-Cal across most metrics. The following table summarizes model performance:
Model	PR-AUC	ROC-AUC	F1-macro	Recall@Top-10%
				
RF+SMOTE-Cal	11.8%	58.8%	51.2%	13.2%
XGB+SMOTE-Cal	13.6%	61.6%	53.1%	17.2%
[Insert Visual 5: Performance Comparison Bar Chart]
[Insert Visual 6: Precision-Recall or Calibration Curve]
5. Novelty and Future Work
This iteration introduces methodological novelty through a time-aware, leak-free split protocol and calibration-first design. Future work includes reintroducing engineered rolling features (means, volatility, spreads, and seasonality) to enhance model performance.
[Insert Visual 7: Project Roadmap (Phase 1 → Phase 2)]
6. Conclusion
The project successfully established a robust baseline for customer churn prediction with calibrated, time-aware models. Despite modest scores, the results are reliable and interpretable. Further feature enhancements are expected to yield significant performance gains in the next iteration.
References
[1] PowerCo Customer Churn Dataset, Kaggle.
[2] Chawla, N.V. et al., 'SMOTE: Synthetic Minority Over-sampling Technique', 2002.
[3] Niculescu-Mizil, A. and Caruana, R. 'Predicting Good Probabilities with Supervised Learning', 2005.
