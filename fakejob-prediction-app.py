import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier


st.title("Fake Job Prediction")
st.write("The app helps in knowing the likelihood of a posted job being fraudulent versus being non-fraudulent. Upon opening the app, the user is prompted to enter few details about the job ad, and based on the XGBoost model used, the user gets to know the likelihood of the job being fraudulent or non-fraudulent. XGBoost is the best model among 6 models we have tested, with the accuracy of 98.6%, precision 92.4%, recall 79.2%.")
# Split into 4 columns, add relevant sliders and button and store in variables
one, two, three, four = st.columns(4)

title = st.text_input("Job title", "")
#one.write(title)

department = st.text_input("Department", "")
#two.write(department)

company_profile = st.text_input("Company profile", "")
#three.write(company_profile)

description = st.text_input("Job description", "")
#four.write(description)

logo = st.selectbox(
    "Do they provide company's logo?",
    ('Yes', 'No'))
if logo == "No":
    st.write("Based on our observation from a large dataset, you should be careful with job ads that don't provide company's logo. However, this doesn't mean all job ads with no logo are fake.")

# Split into 3 columns
one2, two2, three2 = st.columns(3)

requirements = st.text_input("Requirements", "")
#one2.write(requirements)

benefits = st.text_input("Benefits", "")
#two2.write(benefits)

industry = st.text_input("Industry", "")
#three2.write(industry)



# Split into 4 columns, add relevant sliders and button and store in variables
one3, two3, three3, four3= st.columns(4)

required_experience = st.selectbox(
    'Please select the required experience',
    ('Internship', 'Not Applicable', 'Mid-Senior level',
     'Associate', 'Entry level', 'Executive', 'Director'))

st.write('You selected:', required_experience)

required_education = st.selectbox(
    'Please select the required education?',
    ("Bachelor's Degree", "Master's Degree",
     'High School or equivalent', 'Unspecified',
     'Some College Coursework Completed', 'Vocational', 'Certification',
     'Associate Degree', 'Professional', 'Doctorate',
     'Some High School Coursework', 'Vocational - Degree',
     'Vocational - HS Diploma'))

st.write('You selected:', required_education)
 
function = st.selectbox(
    'Please select the job function',
    ('Marketing', 'Customer Service', 'Sales',
    'Health Care Provider', 'Management', 'Information Technology',
    'Other', 'Engineering', 'Administrative', 'Design', 'Production',
    'Education', 'Supply Chain', 'Business Development',
    'Product Management', 'Financial Analyst', 'Consulting',
    'Human Resources', 'Project Management', 'Manufacturing',
    'Public Relations', 'Strategy/Planning', 'Advertising', 'Finance',
    'General Business', 'Research', 'Accounting/Auditing',
    'Art/Creative', 'Quality Assurance', 'Data Analyst',
    'Business Analyst', 'Writing/Editing', 'Distribution', 'Science',
    'Training', 'Purchasing', 'Legal'))

st.write('You selected:', function)

employment_type = st.selectbox(
    'Please select the employment type',
    ('Other', 'Full-time', 'Part-time', 'Contract', 'Temporary'))

st.write('You selected:', employment_type)

#create a DataFrame of the input-------------------------------------------------------------------------------------

trial = pd.DataFrame({'title':title,'department':department,'company_profile':company_profile,'description':description,'requirements':requirements,'benefits':benefits,'employment_type':employment_type, "required_experience":required_experience, "required_education":required_education, "industry":industry, "function":function}, index=[0])

#combine text columns------------------------------------------------------------------------------------------------
text_data=trial.select_dtypes(include="object")
text_col=text_data.columns
trial[text_col]=trial[text_col].replace(np.nan,"")

trial['text'] = ""
for col in text_data.columns:
    trial["text"] = trial["text"] + " " + trial[col]

predictor = trial[['text']]

#clean the text--------------------------------------------------------------------------------------------------------
stop = set(stopwords.words("english"))
def clean(text):
    
    text=text.lower()
    obj=re.compile(r"<.*?>")                     #removing html tags
    text=obj.sub(r" ",text)
    obj=re.compile(r"https://\S+|http://\S+")    #removing url
    text=obj.sub(r" ",text)
    obj=re.compile(r"[^\w\s]")                   #removing punctuations
    text=obj.sub(r" ",text)
    obj=re.compile(r"\d{1,}")                    #removing digits
    text=obj.sub(r" ",text)
    obj=re.compile(r"_+")                        #removing underscore
    text=obj.sub(r" ",text)
    obj=re.compile(r"\s\w\s")                    #removing single character
    text=obj.sub(r" ",text)
    obj=re.compile(r"\s{2,}")                    #removing multiple spaces
    text=obj.sub(r" ",text)
    
    stemmer = SnowballStemmer("english")
    text=[stemmer.stem(word) for word in text.split() if word not in stop]

    porter_stemmer = PorterStemmer()             #defining the object for stemming
    text = [porter_stemmer.stem(word) for word in text]
    wordnet_lemmatizer = WordNetLemmatizer()     #lemmatization
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

#apply clean text function
predictor["text"]=predictor["text"].apply(clean)

#Vectorizer----------------------------------------------------------------------------------------
corpus = predictor["text"]
tfidf  = TfidfVectorizer()
xtrain = pd.read_csv("copyXtrain.csv")
tfidf.fit(xtrain['text'])
corpus = tfidf.transform(corpus)

#Load model---------------------------------------------------------------------------------------

model = XGBClassifier()
model.load_model('xgbmodel.json')

#Predict-------------------------------------------------------------------------------------------
st.subheader("Our XGBoost Model Predicts that this Job ad is: ")

y_predict = model.predict(corpus)
result = y_predict[0]
if(result == 1):
    st.header("Most likely fake!")
elif (result == 0):
    st.header("Most likely genuine!")

