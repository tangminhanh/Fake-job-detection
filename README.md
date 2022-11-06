# IE0005 PROJECT: FAKE JOB PREDICTION

This is a group project for IE0005 - Intro to Data Science & AI module, Sem 1 AY22/23 done by 4 NTU students:
- Pham Thuy Linh (IEM/Y2)
- Tang Minh Anh (EEE/Y2)
- Paramel Jose Paul (EEE/Y2)
- Agrawal Sparsh (EEE/Y2)

GitHub link: https://github.com/ptlinh1803/IE0005-fake-job-detection.git <br>

## Project description
University students and experienced hires are always on the run to find new opportunities. This is often marked by applying to different roles, only to find out that those jobs were fraudulent. This leads to confusion, and most often, results in financial scams also during the application process.
<br><br>
Through this project, we aim to classify jobs as fraudulent versus non-fraudulent based on the dataset, further classifying based on location and other factors such as education, function, industry and required experience. All of this comes to life mostly after screening non-essential, yet important factors such as the steps in hiring process.
<br><br>
<b>Dataset on Kaggle:</b> https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction <br>
This dataset contains 18K job descriptions out of which about 800 are fake. The data consists of both textual information and meta-information about the jobs. The dataset can be used to create classification models which can learn the job descriptions which are fraudulent.

## Project execution plan & result
- Exploratory Data Analysis: explore and compare the attributes of fraudulent and non-fraudulent job ads (Linh, Jose, Sparsh)
- Data cleaning & Prepocessing: use nlkt - Natural Language Toolkit (Minh Anh)
- Build the models: (Linh & Minh Anh)
  + Vectorizing using TfidfVectorizer
  + Oversampling & Undersampling using SMOTE and TomekLinks to balance the train set
  + 6 models:
    * Logistic Regression
    * Random Forest
    * Support Vector Machine
    * XGBoost Classifier
    * KNeighbors Classifier
    * Decision Tree
 - Evaluate models: since want to use minimize False Negative (incorrectly predict fraudulent jobs as non-fraudulent), we pay more attention to Recall and F2 score. When there are more than 1 model with approximately the same recall and F2 score, we take other evaluation metrics (accuracy, prediction, F1 score) in to consideration.
    + Logistic Regression has the best overall performance on the test set: 
      * XGB Accuracy: 98.6%
      * XGB Recall: 79.2%
      * XGB Precision: 92.4%
      * XGB F1 score: 0.853
      * XGB F2 score: 0.815
- Build the streamlit app: deploy the app using Streamlit Cloud and GitHub (Linh, Jose, Sparsh)

## Files
- fakejob-EDA-final-version.ipynb: Exploratory Data Analysis of the dataset
- Models-OFFICIAL-final-version.ipynb: Models exploration
- fake_job_postings.csv: the dataset
- fakejob-prediction-app.py: the streamlit app
- requirements.txt: libraries requirements for the app to run
- xgbmodel.json: extract and save the XGBoost model for the app to use
- copyXtrain.csv: copy of the X_train before vectorization (used again in the streamlit app)

## Application
- <b>Download and run the app locally:</b>
  + Install stramlit: `pip install streamlit`
  + In the cmd, run the command: `streamlit run fakejob-prediction-app.py`
 - <b>Use the app online:</b>
    + Streamlit app link: https://ptlinh1803-ie0005-fake-job-detect-fakejob-prediction-app-og5bti.streamlitapp.com/

## References
- https://analyticsindiamag.com/classifying-fake-and-real-job-advertisements-using-machine-learning/
- https://www.kaggle.com/code/shivamburnwal/nlp-98-acc-eda-with-model-using-spacy-pipeline
- https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/
- https://medium.com/analytics-vidhya/applying-text-classification-using-logistic-regression-a-comparison-between-bow-and-tf-idf-1f1ed1b83640
- https://docs.streamlit.io/library/cheatsheet
- https://stats.stackexchange.com/questions/251871/vectorization-of-data-before-splitting-in-to-test-and-train-with-neural-network
- https://towardsdatascience.com/boosting-techniques-in-python-predicting-hotel-cancellations-62b7a76ffa6c
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
- https://machinelearningmastery.com/fbeta-measure-for-machine-learning/
- https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
- https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/#:~:text=Oversampling%20methods%20duplicate%20or%20create,of%20methods%20are%20used%20together.
