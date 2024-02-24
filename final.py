
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import cv2
import re
import pytesseract
from PIL import Image
import base64
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, accuracy_score,recall_score,classification_report
import seaborn as sns
import pickle
import warnings
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import norm
import plotly.express as px
from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib.colors import ListedColormap
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from spacy import displacy
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.decomposition import TruncatedSVD
SVD = TruncatedSVD(n_components=10)
import warnings
import os
import importlib
from surprise import KNNWithMeans

# warnings.filterwarnings('ignore')
from PIL import Image, ImageFilter, ImageEnhance
from PIL import Image, ImageDraw,ImageChops
sns.set_theme(color_codes=True)

st.set_page_config(page_title="E-commerce Customer Segmentation", layout="wide")

def rear_image(image):
    with open(image, "rb") as image_file:
        encode_str = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encode_str.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

rear_image("bg.jpg")




with st.sidebar:

    
    opt = option_menu("Title",
                      ["Image Processing","Natural Language processing","Exploratory data analysis","Data predictions","Evaluation Metrics","Customer Recomendation"],
                      
                      menu_icon="cast",
                      styles={
                          "container": {"padding":"4!important", "background-color":"dark"},
                          "icon":{"color":"#01A982","font-size":"15px"},
                          "nav-link": {"font-size": "20px", "text-align":"centered"},
                          "nav-link-selected": {"background-color": "red"} })
    


if opt =="Natural Language processing":

    st.title('NLP Detailing...')

    # Download NLTK resources
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # Download spaCy model
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    # Sample data for text classification
    # Your text data
    text_data = [
        ("I love streamlit", "positive"),
        ("Streamlit is easy to use", "positive"),
        ("NLP processing in streamlit is great", "positive"),
        ("Streamlit helps in building interactive web apps", "positive"),
        ("I dislike bugs in streamlit", "negative"),
        ("Streamlit could improve in some areas", "negative"),
        ("NLP can be challenging for beginners", "negative"),
        ("I struggle with streamlit syntax", "negative"),
    ]

    df = pd.DataFrame(text_data, columns=['text', 'label'])

    X_train=df["text"]
    y_train=df["label"]

    

    # Text input
    text_input = st.text_area("Enter text for NLP processing:",max_chars=500)

                                                                        


    # Tokenization
    if ("Tokenization"):
        tokens = word_tokenize(text_input)
        st.text_input("Tokens:", tokens)

    # Stopword Removal
    if ("Stopword Removal"):
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        st.write("Tokens after stopword removal:", filtered_tokens)

    # Number Removal
    if ("Number Removal"):
        filtered_tokens = [word for word in filtered_tokens if not word.isdigit()]
        st.write("Tokens after number removal:", filtered_tokens)

    # Special Character Removal
    if ("Special Character Removal"):
        filtered_tokens = [word for word in filtered_tokens if word.isalnum()]
        st.write("Tokens after special character removal:", filtered_tokens)

    # Stemming
    if ("Stemming"):
        porter_stemmer = PorterStemmer()
        stemmed_tokens = [porter_stemmer.stem(word) for word in filtered_tokens]
        st.write("Tokens after stemming:", stemmed_tokens)

    # Lemmatization
    if ("Lemmatization"):
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        st.write("Tokens after lemmatization:", lemmatized_tokens)

    # Parts of Speech (POS)
    if ("Parts of Speech (POS)"):
        doc = nlp(text_input)
        pos_tags = [(token.text, token.pos_) for token in doc]
        st.write("Parts of Speech:", pos_tags)

    # N-gram
    if ("N-gram"):
        n = st.slider("Select N for N-gram", min_value=2, max_value=5, value=2, step=1)
        ngram_vectorizer = CountVectorizer(ngram_range=(n, n))
        X_ngram = ngram_vectorizer.fit_transform([text_input])
        st.write(f"{n}-gram representation:", X_ngram.toarray())

    # Text Classification
    if ("Text Classification"):
    # Create a pipeline with CountVectorizer and MultinomialNB

        model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = model.predict(X_train)

        # Display evaluation metrics
        accuracy = accuracy_score(y_train, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:\n", classification_report(y_train, y_pred))

    # Sentiment Analysis
    if ("Sentiment Analysis"):
        #Assuming binary sentiment classification (positive and negative)
        sentiment = "Positive" if model.predict([text_input])[0] == "positive" else "Negative"
        st.write(f"Sentiment: {sentiment}")

    # Word Cloud
    if ("Word Cloud"):
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_input)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()

    # Keyword Extraction
    if ("Keyword Extraction"):
        keywords = nlp(text_input).ents
        st.write("Keywords:", [keyword.text for keyword in keywords])

    # Named Entity Recognition (NER)
    if ("Named Entity Recognition (NER)"):
        doc_ner = nlp(text_input)
        ner_displacy = displacy.render(doc_ner, style="ent", page=True)
        st.write(ner_displacy, unsafe_allow_html=True)

        






if opt=="Data predictions":

    st.title('Data Prediction...')

    
    c1,c2,c3,c4 = st.columns(4)
    st.markdown("""
    <style>
        .st-ax {
                background-color: B6FCD5;
        }

        .stTextInput input{
                background-color: dark Green;
        }

        .stNumberInput input{
                background-color: pale blue;
        }
        .stDateInput input{
                background-color: brown;
        }

    </style>
    """,unsafe_allow_html=True)
    with open("model_rf.pkl", "rb") as mf:
        new_model = pickle.load(mf)

    # Input form
        st.markdown("<h1>Data Predictions</h1>",unsafe_allow_html=True)
    with st.form("user_inputs"):
        with st.container():
            count_session = st.slider("session count!",max_value=100.00,min_value=0.00)
            time_earliest_visit = st.slider("Earliest Time Visit",max_value=100.00,min_value=0.00)
            avg_visit_time = st.slider("average visit time",max_value=100.00,min_value=0.00)
            days_since_last_visit = st.slider("days_since_last_visit",max_value=100.00,min_value=0.00)
            days_since_first_visit = st.slider("days_since_first_visit",max_value=100.00,min_value=0.00)
            visits_per_day = st.slider("visits_per_day",max_value=100.00,min_value=0.00)
            bounce_rate = st.slider("bounce_rate",max_value=100.00,min_value=0.00)
            earliest_source = st.slider("earliest_source",max_value=100.00,min_value=0.00)
            latest_source = st.slider("latest_source",max_value=100.00,min_value=0.00)
            earliest_medium = st.slider("earliest_medium",max_value=100.00,min_value=0.00)
            latest_medium = st.slider("latest_medium",max_value=100.00,min_value=0.00)
            earliest_keyword = st.slider("earliest_keyword",max_value=100.00,min_value=0.00)
            latest_keyword = st.slider("latest_keyword",max_value=100.00,min_value=0.00)
            earliest_isTrueDirect = st.slider("earliest_isTrueDirect",max_value=100.00,min_value=0.00)
            latest_isTrueDirect = st.slider("latest_isTrueDirect",max_value=100.00,min_value=0.00)
            num_interactions = st.slider("num_interactions",max_value=100.00,min_value=0.00)
            bounces = st.slider("bounces",max_value=100.00,min_value=0.00)
            time_on_site = st.slider("time_on_site",max_value=100.00,min_value=0.00)
            time_latest_visit = st.slider("time_latest_visit",max_value=100.00,min_value=0.00)

        submit_button = st.form_submit_button(label="Convert / Not Convert")

    # Predict using the model
    if submit_button:
        test_data = np.array([
            [
                count_session, time_earliest_visit, avg_visit_time, days_since_last_visit,
                days_since_first_visit, visits_per_day, bounce_rate, earliest_source,
                latest_source, earliest_medium, latest_medium, earliest_keyword,
                latest_keyword, earliest_isTrueDirect, latest_isTrueDirect, num_interactions,
                bounces, time_on_site, time_latest_visit
            ]
        ])

        # Convert the data to float
        test_data = test_data.astype(float)

        # Make predictions
        predicted = new_model.predict(test_data)[0]
        prediction_proba = new_model.predict_proba(test_data)

        # Display the results
        st.write("Prediction:", predicted)
        st.write("Prediction Probability:", prediction_proba)

if opt=="Evaluation Metrics":

     
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # EDA and Preprocessing Steps

        # Duplicate Removal
        st.write("### Duplicate Removal")
        df1 = df.drop_duplicates()
        st.success("Duplicates removed successfully!")

        # NaN Value Fill
        st.write("### NaN Value Fill")
        df2 = df1.fillna(0)  # You can replace 0 with the desired value
        st.success("NaN values filled successfully!")

        # DateTime Format Conversion
        st.write("### DateTime Format Conversion")
        date_columns = df2.select_dtypes(include=['datetime']).columns
        for col in date_columns:
            df2[col] = pd.to_datetime(df2[col])
        st.success("DateTime Format Conversion completed successfully!")

        # Display DataFrame
        st.dataframe(df2)

        # Summary Statistics
        st.write("### Summary Statistics")
        st.write(df2.describe())


# Feature Importance with Random Forest
        le = LabelEncoder()
        for col in df2.columns:
            if df2[col].dtype == 'object' or df2[col].dtype == 'bool':
                df2[col] = le.fit_transform(df2[col])

        X_train = df2.drop('has_converted', axis=1)
        y_train = df2['has_converted']

       # Plot feature importance
        st.write("### Feature Importance with Random Forest")
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        feature_importances = rf.feature_importances_

        feature_importance_df=pd.DataFrame({
            "Feature":X_train.columns,
            "Impotance":feature_importances
            })
        top_10_features=feature_importance_df.sort_values(by="Impotance",ascending=False).head(10)["Feature"].tolist()
        extra_feature="has_converted"
        df3 = df2[top_10_features + [extra_feature]]
        columns=['count_session','time_earliest_visit','avg_visit_time','days_since_last_visit','days_since_first_visit','visits_per_day','bounce_rate','earliest_source','latest_source','earliest_medium','has_converted']

        # Streamlit code
        st.title('Top 10 Features Importance')
        st.bar_chart(top_10_features)

        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Pie Chart using Feature Importance
        st.write("### Pie Chart using Feature Importance")
        fig, ax = plt.subplots()
        ax.pie(feature_importances, labels=feature_importances, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

        # Random Forest Model Build
        df3 = pd.DataFrame(df3)

         # Drop the 'has_converted' column
        X = df3.drop('has_converted', axis=1)
        y=df3['has_converted']


        # Random Forest Model Build
        model = RandomForestClassifier(n_estimators=50,random_state=42)

        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_predict=rf_model.predict(X_train)
        rf_accuracy = accuracy_score(y_train,rf_predict)
        rf_Precision=precision_score(y_train,rf_predict)
        rf_recall=recall_score(y_train,rf_predict)
        rf_f1=f1_score(y_train,rf_predict)

        # Display Random Forest Model results
        st.write("# Random Forest Model")
        st.write("Accuracy:", rf_accuracy)
        st.write("Precision:", rf_Precision)
        st.write("Recall:", rf_recall)
        st.write("F1_score:", rf_f1)

        # Decision Tree Model Build
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        dt_predict=dt_model.predict(X_train)
        dt_accuracy = accuracy_score(y_train,dt_predict)
        dt_Precision=precision_score(y_train,dt_predict)
        dt_recall=recall_score(y_train,dt_predict)
        dt_f1=f1_score(y_train,dt_predict)

      
        # Display Decision Tree Model results
        st.write("# Decision Tree Model")
        st.write("Accuracy:", dt_accuracy)
        st.write("Precision:", dt_Precision)
        st.write("Recall:", dt_recall)
        st.write("F1_score:", dt_f1)


       
        # KNN Model Build
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)  # Use the X_train, y_train from the first block
        knn_predict=knn_model.predict(X_train)
        knn_accuracy = accuracy_score(y_train, knn_predict)
        knn_Precision=precision_score(y_train,knn_predict)
        knn_recall=recall_score(y_train,knn_predict)
        knn_f1=f1_score(y_train,knn_predict)


         # Display KNN Model results
        st.write("# KNN Model")
        st.write("Accuracy:", knn_accuracy)
        st.write("Precision:",knn_Precision)
        st.write("Recall:", knn_recall)
        st.write("F1_score:",knn_f1)

    # Display results in a table
        results_data = {
            'Model': ['Random Forest', 'Decision Tree', 'KNN'],
            'Accuracy': [rf_accuracy, dt_accuracy, knn_accuracy],
            'Precision': [rf_Precision, dt_Precision, knn_Precision],
            'Recall': [rf_recall, dt_recall, knn_recall],
            'F1_score': [rf_f1, dt_f1, knn_f1]
        }

        results_table = st.table(results_data)
     # Plotly Visualization
        
        fig = px.bar(
            x=['Random Forest', 'Decision Tree', 'KNN'],
            y=[rf_accuracy, dt_accuracy, knn_accuracy],
            labels={'y': 'Accuracy', 'x': 'Models'},
            title='Model Accuracy Comparison'
        )

        st.plotly_chart(fig)



if opt == "Exploratory data analysis":
        

        st.title('EDA Processing...')
        
        class SessionState:
            def __init__(self, **kwargs):
                for key, val in kwargs.items():
                    setattr(self, key, val)


        session_state = SessionState(df=None)

        # Step 1: Load CSV File
        uploaded_file = st.file_uploader("Choose a CSV file",type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            session_state.df = df  # Save the data in the session state

        # Step 2: Display DataFrame
        if session_state.df is not None and st.button("Show DataFrame"):
            st.dataframe(session_state.df)

        if session_state.df is not None:
            st.write("### DataFrame")
            st.dataframe(session_state.df)

        # Droping  Duplicates and NaN Values:
        if session_state.df is not None:
            st.write("### Droping  Duplicates values and NaN Values")
            session_state.df = session_state.df.drop_duplicates()
            session_state.df = session_state.df.dropna()
            st.dataframe(session_state.df)
            st.success("Duplicates and NaN values dropped successfully!")

        if session_state.df is not None:
            st.write("### Information about dataframe:")
            st.text(session_state.df.info())

        if session_state.df is not None:
            st.write("### Summarzing the Dataframe with statistics:")
            st.text(session_state.df.describe())


        # Label Encoding
        if session_state.df is not None:
            st.write("### Label Encoding")
            le = LabelEncoder()
            for col in session_state.df.columns:
                if session_state.df[col].dtype == 'object' or session_state.df[col].dtype == 'bool':
                    session_state.df[col] = le.fit_transform(session_state.df[col])
            st.dataframe(session_state.df)
            st.success("Label Encoding completed successfully!")

        # One-Hot Encoding for categorical columns
        if session_state.df is not None:
            st.write("### One-Hot Encoding")
            categorical_columns = session_state.df.select_dtypes(include=['object']).columns
            session_state.df = pd.get_dummies(session_state.df, columns=categorical_columns)
            st.dataframe(session_state.df)
            st.success("One-Hot Encoding completed successfully!")

        # DateTime Format Conversion
        if session_state.df is not None:
            st.write("### DateTime Format Conversion")
            session_state.df['target_date'] = pd.to_datetime(session_state.df['target_date'])
            st.dataframe(session_state.df)
            st.success("DateTime Format Conversion completed successfully!")

        # Plot Relationship Curve
        if session_state.df is not None:
            st.write("### Plot Relationship Curve")
            sampled_df = pd.DataFrame(session_state.df["avg_visit_time"].sample(min(1000, len(session_state.df))))
            sns.pairplot(sampled_df)
            st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)

        # Detect and Treat Outliers
        if session_state.df is not None:
            st.write("### Detect and Treat Outliers")
            Q1 = session_state.df['transactionRevenue'].quantile(0.25)
            Q3 = session_state.df['transactionRevenue'].quantile(0.75)
            IQR = Q3 - Q1
            session_state.df = session_state.df[~((session_state.df['transactionRevenue'] < (Q1 - 1.5 * IQR)) | (session_state.df['transactionRevenue'] > (Q3 + 1.5 * IQR)))]
            st.dataframe(session_state.df)
            st.success("Outliers detected and treated successfully!")

        # Plot Normalization Curve
        if session_state.df is not None:
            st.write("### Plot Normalization Curve")
            sns.histplot(session_state.df['avg_session_time'], kde=True)
            st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)

        # Treat Skewness
        if session_state.df is not None:
            st.write("### Treat Skewness")
            session_state.df['latest_visit_number'] = np.log1p(session_state.df['latest_visit_number'])
            st.dataframe(session_state.df)
            st.success("Skewness treated successfully!")

        # Calculate Correlation and Plot Heatmap
        if session_state.df is not None:
            st.write("### Correlation Heatmap")
            correlation_matrix = session_state.df.corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
            st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)


        # Box Plot for Outlier Detection
        if session_state.df is not None:
            st.write("### Box Plot for Outlier Detection")
            numeric_columns = session_state.df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                sns.boxplot(x=col, data=session_state.df)
                st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)

        # scatter plot matric
        if session_state.df is not None:
            st.write("### Scatter Plot Matrix")
            sns.set(style="ticks", rc={"figure.autolayout": False})
            sampled_df = pd.DataFrame(session_state.df.sample(min(1000, len(session_state.df))))

            progress_bar = st.progress(0)
            for i in range(len(sampled_df.columns)):
                sns.pairplot(sampled_df, vars=[sampled_df.columns[i]], diag_kind='hist')
                progress_bar.progress((i + 1) / len(sampled_df.columns))

            st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)        



if opt =="Customer Recomendation":


    st.title('Amazon product Recommendation...')
    
    
    
    select = st.multiselect("Select the options", options=("","Recommendation"),default="Recommendation")
    
    
    df=pd.read_csv('Amazon_recommendation.csv',names=['userId', 'productId','rating','timestamp'])

    

    #df.shape

    #df.columns

    electronics_data=df.sample(n=1564896,ignore_index=True)

    del df

    

    #electronics_data.head()

    #electronics_data.info()

    electronics_data.drop('timestamp',axis=1,inplace=True)

    # electronics_data.describe()

    # electronics_data.isnull().sum()

    # electronics_data[electronics_data.duplicated()].shape[0]

    # electronics_data.head()

    st.header("Showing Dataframe: ")

    st.dataframe(electronics_data)

    plt.figure(figsize=(8,4))
    sns.countplot(x='rating',data=electronics_data)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.grid()
    plt.show()

    print('Total rating : ',electronics_data.shape[0])
    print('Total unique users : ',electronics_data['userId'].unique().shape[0])
    print('Total unique products : ',electronics_data['productId'].unique().shape[0])

    no_of_rated_products_per_user = electronics_data.groupby(by='userId')['rating'].count().sort_values(ascending=False)
    no_of_rated_products_per_user.head()

    print('No of rated product more than 50 per user : {} '.format(sum(no_of_rated_products_per_user >= 50)))

    data=electronics_data.groupby('productId').filter(lambda x:x['rating'].count()>=50)

    #data.head()

    no_of_rating_per_product=data.groupby('productId')['rating'].count().sort_values(ascending=False)

    no_of_rating_per_product.head()

    #top 20 product
    no_of_rating_per_product.head(20).plot(kind='bar')
    plt.xlabel('Product ID')
    plt.ylabel('num of rating')
    plt.title('top 20 procduct')
    
    plt.show()

    mean_rating_product_count=pd.DataFrame(data.groupby('productId')['rating'].mean())

    #mean_rating_product_count.head()

    plt.hist(mean_rating_product_count['rating'],bins=100)
    plt.title('Mean Rating distribution')
    plt.show()

    mean_rating_product_count['rating'].skew()

    mean_rating_product_count['rating_counts'] = pd.DataFrame(data.groupby('productId')['rating'].count())

    #mean_rating_product_count.head()

    mean_rating_product_count[mean_rating_product_count['rating_counts']==mean_rating_product_count['rating_counts'].max()]

    print('min average rating product : ',mean_rating_product_count['rating_counts'].min())
    print('total min average rating products : ',mean_rating_product_count[mean_rating_product_count['rating_counts']==mean_rating_product_count['rating_counts'].min()].shape[0])

    plt.hist(mean_rating_product_count['rating_counts'],bins=100)
    plt.title('rating count distribution')
    plt.show()

    sns.jointplot(x='rating',y='rating_counts',data=mean_rating_product_count)
    plt.title('Joint Plot of rating and rating counts')
    plt.tight_layout()
    plt.show()

    plt.scatter(x=mean_rating_product_count['rating'],y=mean_rating_product_count['rating_counts'])
    plt.show()

    print('Correlation between Rating and Rating Counts is : {} '.format(mean_rating_product_count['rating'].corr(mean_rating_product_count['rating_counts'])))

    #Reading the dataset
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(data,reader)

    trainset, testset = train_test_split(surprise_data, test_size=0.3,random_state=42)

    algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
    algo.fit(trainset)

    test_pred=algo.test(testset)

    print("Item-based Model : Test Set")
    accuracy.rmse(test_pred ,verbose=True)

    data2=data.sample(20000)
    ratings_matrix = data2.pivot_table(values='rating', index='userId', columns='productId', fill_value=0)
    #ratings_matrix.head()

    #ratings_matrix.shape

    x_ratings_matrix=ratings_matrix.T
    #x_ratings_matrix.head()

    #x_ratings_matrix.shape


    decomposed_matrix = SVD.fit_transform(x_ratings_matrix)
    #decomposed_matrix.shape

    correlation_matrix = np.corrcoef(decomposed_matrix)
    #correlation_matrix.shape

    User_input = st.text_input("Enter your product Id,")


    st.dataframe(x_ratings_matrix.index)

   
    
    if User_input:

        i=str(User_input)
        product_names=list(x_ratings_matrix.index)
        
        product_id=product_names.index(i)
       
        #st.write(product_id)

        correlation_product_ID = correlation_matrix[product_id]
        #correlation_product_ID.shape

        #correlation_matrix[correlation_product_ID>0.75].shape

        rec = st.button("Get Recommendations")
        

        if rec :

            recommend = list(x_ratings_matrix.index[correlation_product_ID > 0.75])
            #recommend[:20]

            st.title('Top-5 Recommendation!:sunglasses:')

            st.subheader(recommend[:5])

        
if opt == 'Image Processing':
    
    uploadedfile = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploadedfile is not None:
        # Read the image
        image = cv2.imread(uploadedfile.name)
        options = st.multiselect(
        "Select your Image Processing Steps",
        ["Display","Details of an Image",'Preprocess','Image to numpy Array',"Image Shape","Grayscale", "Resize","RGB channel", "Rotate", "crop", "Mirror", "Brightness", "Edge Detection", "'Sharpening", "Mask"],
        
        )
        
        for option in options:



            if option == 'Preprocess':



                    def preprocess_image(image):
                            
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        
                            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                            pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

                            text = pytesseract.image_to_string(thresh)

                            
                            st.write("Extracted Text:")
                            st.write(text)

                            return thresh
                    image = preprocess_image(image)

                    st.image(image, caption="Processed Image", use_column_width=True)
            

            if option == 'Display':
                    image =  np.array(Image.open(uploaded_file))
                    plt.figure(figsize=(8,8))
                    plt.imshow(image)
                    st.image(image)

            if option == 'Details of an Image':
                        st.write('No of dims: ',image.ndim)     # dimension of an image
                        st.write('Img shape: ',image.shape)    # shape of an image
                        st.write('Dtype: ',image.dtype)
                        st.dataframe(image[20, 20])               
                        st.write(image[:, :, 2].min())

            if option == 'Image to numpy Array':
                    image_array = np.array(image)
                    st.write(image_array) 

            if option == "Image Shape":
                image_array = np.array(image)
                i_shape =image_array.shape 
                st.write(i_shape,": Shape of the image.")

            if option == 'Grayscale':
                    
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                    st.write("After converted to gray image Size: ",gray_image.size)
                    gray_image_arr = np.array(gray_image) 
                    st.write("Array of Gray image: ",gray_image_arr,) 
                    st.write("Gray image shape:",gray_image_arr.shape,)
                    st.image(gray_image) 

            if option == 'Resize':

                    def resize_image(image, width):
            # Calculate the aspect ratio
                        aspect_ratio = image.shape[1] / image.shape[0]

                # Calculate the new height based on the aspect ratio
                        height = int(width / aspect_ratio)

                # Resize the image
                        resized_image = cv2.resize(image, (width, height))

                        return resized_image
                    

                    desired_width = st.slider("Select Width for Resize", min_value=50, max_value=800, value=400, step=50)
                    image = resize_image(image, desired_width)
                    st.image(image)


            if option == 'Rotate':
                

                    def rotate_image(image, degree):
                        pil_image = Image.fromarray(image)
                        rotated_image = pil_image.rotate(degree)
                        return np.array(rotated_image)
                    
                    degree = st.slider("Mention the angle, you wanted to rotate!",min_value=0,max_value=360, value=45, step=15)
                    image = rotate_image(image, degree)
                    st.image(image)




            if option == "crop":
                    def crop_image(image, x_start, y_start, x_end, y_end):
                            cropped_image = image[y_start:y_end, x_start:x_end]
                            return cropped_image
                    
                    x_start = st.number_input("X Start:", min_value=0, max_value=image.shape[1], value=0)
                    y_start = st.number_input("Y Start:", min_value=0, max_value=image.shape[0], value=0)
                    x_end = st.number_input("X End:", min_value=0, max_value=image.shape[1], value=image.shape[1])
                    y_end = st.number_input("Y End:", min_value=0, max_value=image.shape[0], value=image.shape[0])
                    image = crop_image(image, int(x_start), int(y_start), int(x_end), int(y_end)) 
                    st.image(image,use_column_width=True) 
            
            
            if option == "Mirror":
                    
                    st.write("Before Mirroring")
                    st.image(image)
                    def mirror_image(image):
                            mirrored_image = cv2.flip(image, 1)  # 1 indicates horizontal flip
                            return mirrored_image
                    
                    image = mirror_image(image)  
                    st.write('After Mirroring')
                    st.image(image,use_column_width=True,)  

            if option == 'Brightness':

                    def adjust_brightness(image, factor):
                            
                            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                            
                            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
                            
                            brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                            return brightened_image  

                    brightness_factor = st.slider("Adjust Brightness", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
                    image = adjust_brightness(image, brightness_factor) 
                    st.image(image,use_column_width=True)   

            if option == 'Edge Detection':

                    def edge_detection(image, low_threshold, high_threshold):
                            
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                            
                            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                        
                            edges = cv2.Canny(blurred, low_threshold, high_threshold)

                            return edges  
                    
                    low_threshold = st.slider("Low Threshold", min_value=0, max_value=255, value=50)
                    high_threshold = st.slider("High Threshold", min_value=0, max_value=255, value=150)
                    image = edge_detection(image, low_threshold, high_threshold)
                    st.image(image,use_column_width=True)

            if option == 'Mask':

                    

                    def apply_mask(image, mask):
                        
                            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                        
                            masked_image = cv2.bitwise_and(image, mask)

                            return masked_image   


                    mask_file = st.file_uploader("Choose a mask file (binary image)", type=["png","jpg"])
                    if mask_file is not None:
                            mask = cv2.imread(mask_file.name, cv2.IMREAD_GRAYSCALE) 


                    image = apply_mask(image, mask) 

                    st.write('masked image')
                    st.image(image)   
            
st.sidebar.image("1.png",use_column_width=True)        