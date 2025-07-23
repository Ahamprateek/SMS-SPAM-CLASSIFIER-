# Importing necessary libraries
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image

# Download NLTK data files (only needed once)
nltk.download('punkt')
nltk.download('stopwords')

# Initializing the stemmer (to reduce words to their root form)
ps = PorterStemmer()

# ----------------------------
# 1. Text Preprocessing Function
# ----------------------------
def transform_text(text):
    # Convert all characters to lowercase
    text = text.lower()

    # Tokenize: Break sentence into words
    text = nltk.word_tokenize(text)

    # Remove special characters and punctuation
    text = [word for word in text if word.isalnum()]

    # Remove stopwords (common useless words) and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Apply stemming: reduce words like 'running', 'ran' -> 'run'
    text = [ps.stem(word) for word in text]

    # Join back to string
    return " ".join(text)

# ----------------------------
# 2. Load Trained Model and Vectorizer
# ----------------------------
# Load the TF-IDF vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Load the trained machine learning model (e.g., Naive Bayes)
model = pickle.load(open('model.pkl', 'rb'))

# ----------------------------
# 3. Streamlit Web App UI
# ----------------------------
# Set page title and layout
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")

# Load and display image at the top (cool visual)
image = Image.open("goku.png")
st.image(image, caption="Black Goku", use_container_width=True)  # ✅ No deprecated warning

# Main Title
st.title("📩 SMS Spam Classifier")

# Text input box
input_sms = st.text_area("Enter the message you want to classify 👇")

# Predict button
if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠ Please enter a message before predicting.")
    else:
        # ----------------------------
        # 4. Predict Spam or Ham
        # ----------------------------

        # Step 1: Preprocess the text
        transformed_sms = transform_text(input_sms)

        # Step 2: Vectorize the text (convert to numbers)
        vector_input = tfidf.transform([transformed_sms])

        # Step 3: Make prediction
        result = model.predict(vector_input)[0]

        # Step 4: Display result
        if result == 1:
            st.error("🚫 This message is *Spam*!")
        else:
            st.success("✅ This message is *Not Spam*.")

# Optional footer (for college or GitHub credits)
st.markdown("---")
st.caption("Built with ❤ using Python & Streamlit")