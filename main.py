import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Performance ---
@st.cache_resource
def load_keras_model():
    """Load the pre-trained Keras model from the .h5 file."""
    try:
        return load_model('bilstm_imdb_model.h5')
    except Exception as e:
        st.error(f"Error loading the Keras model: {e}")
        return None

@st.cache_resource
def load_word_index():
    """Load the IMDB dataset's word index."""
    return imdb.get_word_index()

# --- Load Model and Data ---
model = load_keras_model()
word_index = load_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# --- Helper Functions ---
def decode_review(encoded_review):
    """Decodes an encoded review back to human-readable text."""
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    """
    Preprocesses the user's text input to be compatible with the model.
    - Converts text to lowercase and splits into words.
    - Encodes words to integers using the word index.
    - Handles out-of-vocabulary words.
    - Pads the sequence to a fixed length (maxlen=500).
    """
    # The vocabulary size the model was trained on (e.g., top 10,000 words)
    vocab_size = 10000 
    words = text.lower().split()
    
    encoded_review = []
    for word in words:
        # Get the index, default to 2 (unknown word token) if not found.
        index = word_index.get(word, 2)
        # **FIX**: Check if the word's index is within the model's vocabulary.
        if index < vocab_size:
            # We add 3 because indices 0-2 are reserved for padding, start, and unknown.
            encoded_review.append(index + 3)
        else:
            # If the word is out of vocabulary, treat it as an unknown word.
            encoded_review.append(2) # '?' token index
            
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# --- Streamlit App Interface ---

# Title and Subheader
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below, and the AI will determine if it's **Positive** or **Negative**.")

# --- Sidebar for Input ---
st.sidebar.header("Enter Your Review")
user_input = st.sidebar.text_area("Movie Review", height=200, placeholder="e.g., 'The movie was fantastic! The acting was superb and the plot was gripping.'")

if st.sidebar.button('Classify Sentiment', type="primary", use_container_width=True):
    if model and user_input:
        # Show a spinner while processing
        with st.spinner('🧠 Analyzing sentiment...'):
            # Preprocess the input
            preprocessed_input = preprocess_text(user_input)

            # Make a prediction
            prediction = model.predict(preprocessed_input)
            sentiment_score = prediction[0][0]

            # Determine sentiment
            is_positive = sentiment_score > 0.5
            sentiment = 'Positive' if is_positive else 'Negative'
            emoji = '👍' if is_positive else '👎'

        # --- Display the Result ---
        st.header("Analysis Result")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Sentiment",
                value=f"{sentiment} {emoji}",
            )

        with col2:
            st.metric(
                label="Confidence Score",
                value=f"{sentiment_score:.2%}",
                help="This score represents the model's confidence in its prediction. Scores > 50% are classified as Positive."
            )

    elif not user_input:
        st.warning("Please enter a movie review in the sidebar to classify.")

else:
    st.info("Enter a review in the sidebar and click 'Classify Sentiment' to see the magic!")

