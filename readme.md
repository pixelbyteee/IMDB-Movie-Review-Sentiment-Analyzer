# 🎬 IMDB Movie Review Sentiment Analyzer

This project is a web application that uses a deep learning model to perform real-time sentiment analysis on movie reviews. Built with Streamlit and powered by a Bidirectional LSTM model, the app can classify any given text as either **Positive** or **Negative**.

This application serves as an end-to-end demonstration of building, training, and deploying a sophisticated Natural Language Processing (NLP) model.

## 🚀 Live Demo

You can try the live, deployed application here:  
👉 **[Click Here to Access the Web App](https-your-streamlit-app-link-here)**

## 📸 App Screenshot

![alt text](<Screenshot 2025-09-04 002137.png>)

---

## ✨ Features

* **Real-Time Sentiment Analysis**: Instantly classifies movie reviews as positive or negative.
* **Interactive Web Interface**: A clean and user-friendly interface built with Streamlit for easy interaction.
* **Advanced Deep Learning Model**: Utilizes a Bidirectional LSTM (Long Short-Term Memory) network, which provides a high degree of accuracy by understanding the context of words from both forward and backward directions in a sentence.
* **Performance Optimized**: Uses Streamlit's caching (`@st.cache_resource`) to load the TensorFlow/Keras model and tokenizer only once, ensuring fast and efficient performance.
* **Robust Preprocessing**: Handles out-of-vocabulary words to prevent errors and ensure the model can process any user input.

---

## 🛠️ Technology Stack

* **Backend & ML**: Python, TensorFlow, Keras, Scikit-learn, Pandas
* **Frontend Web App**: Streamlit
* **Version Control**: Git & GitHub

---

