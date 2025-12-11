# Automated-Resume-Classification-System
This project is a Document Classification System that extracts text from resumes, trains multiple ML models, and deploys the best one through a Streamlit app. It classifies resumes into predefined categories and supports single, text-based, and batch predictions, making screening faster and more efficient.

.

**📄 Document Classification System**
Automated Resume Categorization Using Machine Learning

This project builds a complete, end-to-end resume classification pipeline that extracts text from DOCX/PDF files, performs exploratory data analysis, trains multiple ML models, evaluates them, and finally serves predictions through a beautiful Streamlit web application.

The system is designed for HR teams and organizations that handle large volumes of resumes and want to automate the filtering process with speed and accuracy.

**⭐ Key Features**
1. Automatic Text Extraction
Extracts readable text from DOCX and PDF files directly from a ZIP dataset.
Handles multiple resume categories like Peoplesoft, React Developer, SQL Developer, and Workday.
Code reference: Data Extraction Script 
**Code reference:1_data_extraction**

**2. Exploratory Data Analysis (EDA)**
Generates insights such as:
-> Category distribution
-> Text length analysis
-> Word clouds
-> Keyword extraction
-> Resume structural feature detection
-> Provides visuals to understand dataset quality and category balance.
-> **Code reference: EDA Script ----> 2_eda_analysis (1)**

**3. Multi-Model Training & Evaluation**
The training pipeline builds and compares multiple ML models:
1. Naive Bayes
2. Logistic Regression
3. SVM
4. Random Forest
5. Gradient Boosting
6. XGBoost
6. LightGBM
7. CatBoost (if available)

The system:
-> Cleans text
-> Vectorizes using TF-IDF
-> Trains models
-> Computes Accuracy, Precision, Recall, F1
-> Performs 5-fold cross-validation
-> Selects the best model
-> Saves model artifacts (.pkl) and metadata (model_info.json)
->** Code reference: Model Training Script **

**3_model_training**

**4. Streamlit Web Application**

A fully interactive UI that allows:

🔹 Upload Resume (PDF/DOCX)

Automatically extract text and classify into the correct category.

🔹 Text Input Mode

Paste raw resume text for instant prediction.

🔹 Batch Processing Mode

Upload multiple files and download results as CSV.
==================================================
The app visualizes:
- Predicted category
- Confidence score
- Probability distribution
- Batch summary stats
- **Code reference: app.py **

app

**🧠 Technology Stack**

- Backend / ML
- Python
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost (optional)
- NLTK for preprocessing
- TF-IDF for vectorization
- Frontend
- Streamlit
- Plotly
- Utilities
- python-docx
- PyPDF2
- joblib
- pandas, numpy

**Project Structure**
├── 1_data_extraction.py      # Extracts text from ZIP resumes
├── 2_eda_analysis.py         # Exploratory data analysis & visualizations
├── 3_model_training.py       # Model training, evaluation, saving best model
├── app.py                    # Streamlit web app
├── extracted_documents.csv   # Dataset generated after extraction
├── best_model.pkl            # Best-performing ML model
├── tfidf_vectorizer.pkl      # Saved vectorizer
├── label_encoder.pkl         # Saved label encoder
├── model_info.json           # Model metadata
└── README.md                 # (You will paste this description here)

**Preview**

Upload Resume (PDF/DOCX)
- Automatically extract text and classify into the correct category.
![image alt](https://github.com/Kpkan003/Automated-Resume-Classification-System/blob/0ee47e4be52628f61a524f744058843c8c5feb98/resume%20cls_1.png)
- File is loaded
![image alt](https://github.com/Kpkan003/Automated-Resume-Classification-System/blob/f03e8b820b9b4c3ca95a10261bb4995e1658820a/resume%20cls_2.png)
- Classification result Report
![image alt](https://github.com/Kpkan003/Automated-Resume-Classification-System/blob/f03e8b820b9b4c3ca95a10261bb4995e1658820a/resume%20cls_3.png)
