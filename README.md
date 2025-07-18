# FUTURE_DS_03
College Event Feedback Analysis – Internship Project(Using NLP + ML + PowerBI)





https://github.com/user-attachments/assets/0d63af60-c39d-447d-9b50-d9e82ee7cde1

link : https://colleegefeedback.streamlit.app/

https://github.com/user-attachments/assets/cc45255d-0eaa-45ce-a716-55f8f5e2e425


---

````markdown
# 📊 Student Feedback Sentiment Analysis & Event Insights Dashboard 🚀






https://github.com/user-attachments/assets/11d79c51-8221-4a06-ad21-3e4a96803097




## 🔍 Project Overview

This project aims to analyze student feedback from campus events (e.g., tech fests, workshops, seminars) using both **structured (ratings)** and **unstructured (text comments)** data. We harness the power of **Python (Google Colab)**, **Advanced NLP**, **Machine Learning**, **Power BI**, and **Streamlit Web App** to turn feedback into **actionable insights**.

---

# ✨ College Event Feedback Analysis (Insight)

## 🔍 Project Overview

Use data science to improve campus life! Learn how to turn student feedback into actionable insights using real-world tools like Google Colab, pandas, and TextBlob — no coding background needed.

College events like tech fests, workshops, and cultural activities collect feedback — but are we using it meaningfully?

In this project, we analyze text and rating-based feedback submitted by students after attending campus events. We work with Google Forms data (CSV) and use Natural Language Processing (NLP) to understand satisfaction levels and identify areas for improvement.

---

## 🎯 What You’ll Do

- ✅ Clean and prepare feedback data (from a Google Form export)
- ✅ Analyze ratings (1–5 scale) to find patterns of satisfaction
- ✅ Use NLP tools to score sentiment in comments (positive/neutral/negative)
- ✅ Visualize trends with beautiful charts and graphs
- ✅ Suggest improvements for future events

---

## 🧠 Skills You’ll Gain

- Data cleaning & preparation with pandas  
- Sentiment analysis using TextBlob or VADER  
- Creating bar charts, pie charts, word clouds for reports  
- Interpreting survey data to help make real decisions  
- Working in Google Colab (no software installation!)

---

## 🛠 Tools & Libraries

| Tool           | Purpose                     |
|----------------|-----------------------------|
| Google Colab   | Online coding (no setup)    |
| pandas         | Data manipulation           |
| seaborn/matplotlib | Visualization          |
| TextBlob / VADER | Sentiment analysis (NLP) |

---

## 🗂️ Sample Dataset (CSV format)

Use any of these or simulate your own:

- 🔗 Student Feedback Survey Responses  
- 🔗 Student Satisfaction Survey  

Or collect real feedback from Google Forms:

1. Ask students to rate and comment after an event.
2. Export responses as CSV.
3. You're ready to analyze!

---

## 📊 Example Insights You Can Find

- ✅ Top 3 events with highest satisfaction
- ✅ Most common complaints (via word cloud)
- ✅ Correlation between ratings and event type (workshop vs seminar)
- ✅ Which departments hosted the most-liked events

---

## 📁 Final Deliverable

- ✅ A clean, well-commented Jupyter Notebook (or Colab link)
- ✅ A mini-report/dashboard with:
  - Graphs of ratings
  - Sentiment analysis summary
  - Key recommendations for event organizers

---

## 📌 Key Insights

### 🏆 Top 3 Events with Highest Satisfaction

- **FYBA** – Average Score: `4.55`  
- **MSc Analytical Chemistry Sem I** – Average Score: `4.53`  
- **TYBSc** – Average Score: `4.52`

These courses consistently received highly positive feedback from students, indicating strong engagement, effective instruction, or well-organized content.

---

### 💬 Most Common Complaints (from Word Cloud)

**Top complaint terms with frequency:**

- `"Average"` — 16 times  
- `"Satisfied"`, `"Teaching"`, `"Method"` — 13 times  
- `"Expected"`, `"Felt"`, `"Boring"` — 9 times  
- `"Confusing"`, `"Session"`, `"Found"` — 5 times  

**Insights:**  
Students were concerned about **unclear content**, **low interactivity**, and **fast pacing**. This shows a need to make content more **engaging**, **clear**, and **well-paced**.

---

### 📈 Rating vs Event Type (Workshop vs Seminar)

- **Seminars** received slightly higher average scores and sentiment scores:
  - Average Score: `3.85` vs `3.81`
  - Percentage Score: `76.96%` vs `76.22%`
  - Compound & Polarity sentiment: marginally better for Seminars

**Conclusion:**  
Seminars performed slightly better than workshops, but both were generally well-received.

---

### 🏅 Most-Liked Departments

Departments with highest satisfaction:

- **Information Technology** — Avg: `4.35`
- **Banking and Insurance** — Avg: `4.35`
- **Arts** — Avg: `4.34`, Compound: `0.426`

Departments with lowest scores:

- **Data Science** — Avg: `3.05`, Percentage: `61.00%`
- **Physics** and **Food** also showed lower engagement

**Conclusion:**  
High-performing departments provide engaging sessions; underperforming ones may need to reassess content and delivery style.

---

## 🔍 Key Findings Summary

### Event Type vs Ratings

| Metric            | Seminar   | Workshop  |
|-------------------|-----------|-----------|
| Average Score     | 3.85      | 3.81      |
| Percentage Score  | 76.96%    | 76.22%    |
| Sentiment Scores  | Higher    | Lower     |

### Department-wise Performance

- ⭐️ Top Departments: **IT**, **Banking**, **Arts**
- ⚠️ Low Engagement: **Data Science**, **Food**, **Physics**

### Sentiment Analysis (Text Feedback)

- Majority of comments were **positive**
- Negative feedback focused on:
  - Lengthy sessions
  - Lack of interactivity
  - Overly technical content

---

## ✅ Key Recommendations for Event Organizers

### 1. 📚 Prioritize Seminar-Style Delivery

Seminars had slightly better reception.

**Recommendation:** Blend workshop elements into seminars (e.g., add interactive demos after talks).

---

### 2. 🧪 Replicate Best Practices from Top Departments

**IT, Arts, Banking & Insurance** consistently rated highly.

**Recommendation:**  
Study their approach—strong speakers, relevant topics, engaging formats—and use them as a model for other departments.

---

### 3. 📉 Address Low-Scoring Departments

**Data Science and Food** scored the lowest.

**Recommendation:**

- Gather more feedback
- Improve speakers and content alignment
- Conduct smaller trial sessions

---

### 4. 💬 Enhance Interactive Elements

Workshops should include:

- Breakout sessions  
- Live demos  
- Polls and quizzes  

---

### 5. ⏱ Shorten Lengthy Sessions

Long, dense sessions led to boredom.

**Recommendation:**

- Keep sessions within **60–90 mins**
- Include breaks
- Focus on interaction

---

### 6. 📊 Use Feedback Data for Continuous Improvement

Track trends over time using:

- Ratings
- Sentiment scores
- Department-wise dashboards

---

### 7. 🎤 Train Event Speakers

Offer training on:

- Audience engagement  
- Clear content delivery  
- Visual aids and examples

---

### ✅ Summary Recommendations

- Encourage interactive sessions
- Balance theory with practice
- Improve clarity and relevance
- Optimize session length
- Train facilitators effectively
- Collect event-specific feedback
- Leverage dashboards for tracking progress

---

🎉 **Thank you for exploring the project!** Feedback-driven event planning is the future of student engagement.


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
code = '''import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and vectorizer
model = joblib.load("sentiment_rf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit page settings
st.set_page_config(
    page_title="🎓 Student Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="auto"
)

# Sidebar Info
with st.sidebar:
    st.title("📘 About")
    st.markdown(\"\"\"
    This app analyzes **student feedback** and classifies the sentiment as:

    - 🟢 Positive
    - 🟡 Neutral
    - 🔴 Negative

    Built using **Streamlit**, **scikit-learn**, and **NLP (TF-IDF + RandomForest)**.
    \"\"\")

# Main Header
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📊 Student Feedback Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)
st.write("")

# Input Box
user_input = st.text_area("✍️ Enter Student Comment Below:")

# Predict Button
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a comment before predicting.")
    else:
        input_vector = vectorizer.transform([user_input])
        pred = model.predict(input_vector)[0]

        sentiment_map = {0: "🔴 Negative", 1: "🟡 Neutral", 2: "🟢 Positive"}
        sentiment_label = sentiment_map.get(pred, str(pred))

        # Result Card
        st.markdown("---")
        st.markdown(f"<h3 style='text-align: center;'>Prediction:</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<h2 style='text-align: center; color: #2E8B57;'>{sentiment_label}</h2>",
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.success("✅ Sentiment analysis complete.")

# Footer
st.markdown(
    "<hr><div style='text-align: center;'>Made with ❤️ by Gouthum • Future Interns Task 3</div>",
    unsafe_allow_html=True
)
'''

# Save to a Python file in Colab
with open("app.py", "w") as f:
    f.write(code)

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

### 🧑‍🏫 Workshops
- Workshops consistently received the **highest average ratings** across all feedback metrics:
  - `Average_Score`
  - `Compound` (VADER)
  - `Polarity` (TextBlob)
  - `Percentage`

---

### 🔍 Sentiment Analysis (TextBlob & VADER)
- **Sentiment Breakdown** from `Cleaned_Comments`:
  - 🟢 **65% Positive**
  - ⚪ **20% Neutral**
  - 🔴 **15% Negative**
- `Compound` scores from VADER aligned well with `Percentage` ratings and `Average_Score`.

---

### 💡 Text Preprocessing & NLP
- **Text Normalization** included:
  - Lowercasing
  - Removal of punctuation, stopwords, and special characters
  - Tokenization and Lemmatization
- **TF-IDF Vectorization**:
  - Applied to `Cleaned_Comments` for feature extraction.
  - Used in machine learning model training and sentiment quantification.

---

### Most-Liked Departments Insights:

Top-performing departments in terms of feedback are:

Information Technology (4.35),

Banking and Insurance (4.35), and

Arts (4.34), which also has the highest compound sentiment (0.426).

Departments with relatively low engagement include:

Data Science, with the lowest average score (3.05) and lowest percentage (61.00%).

This indicates a gap in satisfaction and potentially lower engagement or alignment of events with the Data Science department's expectations.

Conclusion:

Departments like Information Technology, Banking and Insurance, and Arts show the highest satisfaction levels, while departments like Data Science, Physics, and Food show the lowest feedback scores, indicating a need to reevaluate the event relevance or execution for those departments.

---

### ❌ Common Feedback Issues
Most Common Complaints:
average (16 times)
satisfied (13 times)
teaching (13 times)
method (13 times)
expected (9 times)
felt (9 times)
boring (9 times)
found (5 times)
session (5 times)
confusing (5 times)

---

### 🤖 Machine Learning Models
- Models Developed:
  - ✅ **Logistic Regression**
  - ✅ **Random Forest Classifier**
- **Input Features**:
  - `TF-IDF` feature vectors from `Cleaned_Comments`
- **Target Labels**:
  - Derived from sentiment polarity score buckets (Positive/Neutral/Negative)
- **Best Model**:
  - Saved using `joblib` and deployed in the Streamlit app

---

### 📊 Power BI Dashboard
- Developed with advanced **DAX measures** and **Power Query** transformations.
- Interactive features:
  - ✅ Dynamic filtering by Event Type, Department, and Sentiment
  - ✅ Drill-through analysis by department
  - ✅ KPI cards showing:
    - Avg. Rating
    - Positive/Negative Sentiment %
    - Event Count
  - ✅ Slicers for user-driven exploration
  - ✅ Bar charts, donut charts, line charts for:
    - Sentiment trends
    - Participation volume
    - Rating distribution

---

### 🧠 Correlation Analysis
- **Positive Correlation** observed between:
  - `Average_Score` and `Polarity`
  - `Average_Score` and `Compound`
  - `Average_Score` and `Percentage`
- **Workshop events** showed the **strongest correlations**, indicating alignment between participant perception and actual sentiment scores.

---

### ⚙️ Power BI Formulas
- **DAX Measures Used**:
  ```DAX
  ## 📊 Power BI DAX Formulas Used

All the DAX formulas listed below were implemented in the `Cleaned_Student_Satisfaction_Surveys` table to build dynamic visuals, KPIs, and insights for the student satisfaction survey analysis.

---

### ✅ `Average Rating (2 Decimal Precision)`

```DAX
Average Rating = 
ROUND(
    AVERAGE(Cleaned_Student_Satisfaction_Surveys[Total Rating]),
    2
)
## 📊 Power BI DAX Formulas Used

All the DAX formulas listed below were implemented in the `Cleaned_Student_Satisfaction_Surveys` table to build dynamic visuals, KPIs, and insights for the student satisfaction survey analysis.

---
---

### ✅ `Positive Feedback Count`

```DAX
Positive Feedback = 
CALCULATE(
    COUNTROWS(Cleaned_Student_Satisfaction_Surveys),
    Cleaned_Student_Satisfaction_Surveys[Sentiment_Label] = "Positive"
)

---

---

Negative Feedback = 
CALCULATE(
    COUNTROWS(Cleaned_Student_Satisfaction_Surveys),
    Cleaned_Student_Satisfaction_Surveys[Sentiment_Label] = "Negative"
)

----

---

Neutral Feedback = 
CALCULATE(
    COUNTROWS(Cleaned_Student_Satisfaction_Surveys),
    Cleaned_Student_Satisfaction_Surveys[Sentiment_Label] = "Neutral"
)

---


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

**👨‍💻 Gouthum Kharvi**
Data Scientist | Python & NLP Enthusiast | Streamlit + Power BI Developer


---

```


