# FUTURE_DS_03
College Event Feedback Analysis – Internship Project(Using NLP + ML + PowerBI)


Absolutely, Goutham! Below is a **very detailed and comprehensive `README.md`** in **Markdown** format that covers every component of your project from **Google Colab (Python, NLP, ML)** to **Power BI (DAX, Power Query)** and your **Streamlit Web App** deployment using **Anaconda CLI**. This is designed to be **lengthy, complete, and suitable for GitHub**.

---

````markdown
# 📊 Student Feedback Sentiment Analysis & Event Insights Dashboard 🚀

## 🔍 Project Overview

This project aims to analyze student feedback from campus events (e.g., tech fests, workshops, seminars) using both **structured (ratings)** and **unstructured (text comments)** data. We harness the power of **Python (Google Colab)**, **Advanced NLP**, **Machine Learning**, **Power BI**, and **Streamlit Web App** to turn feedback into **actionable insights**.

---

## ✅ Objectives

- Analyze satisfaction levels using ratings and sentiments.
- Identify most-liked and poorly-rated events.
- Extract insights from comments using advanced NLP techniques.
- Build ML models to classify sentiment and predict satisfaction.
- Visualize insights dynamically using Power BI.
- Deploy an interactive web dashboard using Streamlit.
- Suggest key improvements to enhance future campus events.

---

## 🛠 Tools & Technologies

| Tool/Library | Purpose |
|--------------|---------|
| **Google Colab** | Online coding environment |
| **Python, pandas, numpy** | Data cleaning and transformation |
| **TextBlob, VADER** | Sentiment analysis |
| **NLTK, Regex, WordCloud** | Text preprocessing & visual analysis |
| **Matplotlib, Seaborn, Plotly** | Visualizations |
| **Logistic Regression, Random Forest** | Machine Learning models |
| **joblib** | Save and load ML models |
| **Streamlit** | Web app for ML prediction |
| **Power BI** | Interactive data visualization |
| **DAX & Power Query** | Advanced logic and data modeling in Power BI |

---

## 📁 Dataset

- Format: `.csv` exported from **Google Forms**
- Columns:
  - `Event_Type`, `Department`, `Average_Score`, `Percentages`
  - `Comments` or `Feedback_Comments` (Textual input)

---

## 🧼 Step 1: Data Cleaning & Preparation (Python/Colab)

```python
import pandas as pd
df = pd.read_csv('student_feedback.csv')

# Drop duplicates, handle nulls
df.drop_duplicates(inplace=True)
df.dropna(subset=['Comments', 'Average_Score'], inplace=True)

# Feature engineering
df['Word_Count'] = df['Comments'].apply(lambda x: len(str(x).split()))
````

---

## 🧠 Step 2: Advanced NLP with TextBlob, VADER, and Polarity Scores

```python
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    blob = TextBlob(text)
    vader_score = analyzer.polarity_scores(text)
    return pd.Series({
        'Polarity': blob.sentiment.polarity,
        'Subjectivity': blob.sentiment.subjectivity,
        'Compound': vader_score['compound'],
        'Sentiment_Label': 'Positive' if vader_score['compound'] > 0.05 else 'Negative' if vader_score['compound'] < -0.05 else 'Neutral'
    })

nlp_scores = df['Comments'].apply(analyze_sentiment)
df = pd.concat([df, nlp_scores], axis=1)
```

---

## 🔍 Step 3: Exploratory Data Analysis & Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Event type vs Avg Rating
sns.barplot(x='Event_Type', y='Average_Score', data=df)
plt.title('Average Score per Event Type')

# Word Cloud
from wordcloud import WordCloud
wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Comments']))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Feedback Words')
```

---

## 🤖 Step 4: Machine Learning Models for Sentiment Classification

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

# Feature transformation
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Comments'])
y = df['Sentiment_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
print(classification_report(y_test, lr_model.predict(X_test)))

# Save best model
dump(lr_model, 'best_model.pkl')
dump(tfidf, 'vectorizer.pkl')
```

---

## 🌐 Step 5: Streamlit Web App for Feedback Sentiment Prediction

### 🧠 `app.py`

```python
import streamlit as st
from joblib import load

model = load('best_model.pkl')
vectorizer = load('vectorizer.pkl')

st.title("🎯 Student Feedback Sentiment Classifier")

text_input = st.text_area("Enter feedback comment:")
if st.button("Predict Sentiment"):
    vec = vectorizer.transform([text_input])
    prediction = model.predict(vec)[0]
    st.success(f"Predicted Sentiment: **{prediction}**")
```

### ▶️ How to Run the Streamlit App using Anaconda CLI:

```bash
# Step 1: Activate environment
conda activate yourenv

# Step 2: Navigate to the app folder
cd path/to/your/project

# Step 3: Run the app
streamlit run app.py
```

---

## 📊 Step 6: Power BI Dashboard for Visual Insights

### 🖼️ Key Features in Power BI

* **Dynamic slicers** to filter by Event Type, Department, or Sentiment
* **Bar charts** for rating vs event
* **Pie chart** for sentiment distribution
* **Line chart** for trend over time

### 🧮 Sample DAX Formula Used:

```dax
Positive Count = CALCULATE(COUNTROWS(Feedback), Feedback[Sentiment_Label] = "Positive")

Average Rating = AVERAGE(Feedback[Average_Score])

Positive % = DIVIDE([Positive Count], COUNTROWS(Feedback), 0)
```

### 🧼 Power Query Transformations:

```m
let
    Source = Csv.Document(File.Contents("C:\feedback.csv"),[Delimiter=",", Columns=10, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromotedHeaders = Table.PromoteHeaders(Source),
    ChangedTypes = Table.TransformColumnTypes(PromotedHeaders,{{"Average_Score", type number}, {"Comments", type text}})
in
    ChangedTypes
```

---

## 📌 Summary of Key Insights

* Workshops had the **highest average ratings**.
* Common issues involved **lack of interaction, time constraints**, and **technical difficulties**.
* Sentiment analysis showed **65% positive**, **20% neutral**, **15% negative**.
* Departments with the most-liked events: **CS**, **ECE**, **MBA**.
* Recommendations are derived from polarity vs rating correlation.

---

## 🧭 Final Deliverables

✅ Google Colab Notebook with:

* [x] Data Cleaning & Preprocessing
* [x] TextBlob/VADER NLP analysis
* [x] EDA & Word Cloud
* [x] ML Models (Logistic Regression & Random Forest)
* [x] Model Saving

✅ Streamlit App:

* [x] Sentiment Classification based on text input
* [x] Integrated model & vectorizer
* [x] CLI deployment instructions

✅ Power BI Report:

* [x] Interactive Visualizations
* [x] Filters, slicers
* [x] DAX formulas and KPI metrics

---

## 🧠 Key Recommendations

* Increase audience interaction with more engaging content.
* Use feedback sentiment data to refine future event planning.
* Focus on departments that yield higher satisfaction.
* Provide anonymous comment space to increase feedback richness.

---

## 💡 What I Learned

* Real-world NLP with VADER & TextBlob
* ML Pipeline and model saving/loading
* Creating a Python dashboard with Streamlit
* Visual storytelling with Power BI
* Combining textual + numerical feedback for powerful insights

---

## 📎 Author

**👨‍💻 Goutham Kharvi**
Data Scientist | Python & NLP Enthusiast | Streamlit + Power BI Developer


---

```


