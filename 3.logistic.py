import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
import re
from rake_nltk import Rake
import pickle
import streamlit as st
import numpy as np
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
pickle_in = open("C:\\Users\\NITESH\\Downloads\\model.pkl", "rb") 
model = pickle.load(pickle_in)
pickle_in = open("C:\\Users\\NITESH\\Downloads\\vectorizer.pkl", "rb")
vectorizer = pickle.load(pickle_in)
st.title('Text Analysis\n', )
st.header("Sentiment Analysis")
st.subheader("Please enter the review to analyze")

input_text = st.text_area("Enter review", height=50)

option = st.sidebar.selectbox('Navigation',['Sentiment Analysis','Keywords','Word Cloud'])
st.set_option('deprecation.showfileUploaderEncoding', False)
if option == "Sentiment Analysis":
    
    
    
    if st.button("Predict sentiment"):
        st.write("Number of words in Review:", len(input_text.split()))
        wordnet=WordNetLemmatizer()
        text=re.sub('[^A-za-z0-9]',' ',input_text)
        text=text.lower()
        text=text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        pickle_in = open("C:\\Users\\NITESH\\Downloads\\model.pkl", "rb") 
        model = pickle.load(pickle_in)
        pickle_in = open("C:\\Users\\NITESH\\Downloads\\vectorizer.pkl", "rb") 
        vectorizer = pickle.load(pickle_in)
        transformed_input = vectorizer.transform([text])
        
        if model.predict(transformed_input) == -1:
            st.write("Input review has Negative Sentiment.:sad:")
        elif model.predict(transformed_input) == 1:
            st.write("Input review has Positive Sentiment.:smile:")
        else:
            st.write(" Input review has Neutral Sentiment.😐")
         
elif option == "Keywords":
    st.header("Keywords")
    if st.button("Keywords"):
        
        r=Rake(language='english')
        r.extract_keywords_from_text(input_text)
        phrases = r.get_ranked_phrases()
        st.write("These are the *keywords* causing the above sentiment:")
        for i, p in enumerate(phrases):
            st.write(i+1, p)
elif option == "Word Cloud":
    st.header("Word cloud")
    if st.button("Generate Wordcloud"):
        wordnet=WordNetLemmatizer()
        text=re.sub('[^A-za-z0-9]',' ',input_text)
        text=text.lower()
        text=text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        wordcloud = WordCloud().generate(text)
        plt.figure(figsize=(40, 30))
        plt.imshow(wordcloud) 
        plt.axis("off")
        
        st.pyplot()