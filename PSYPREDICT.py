import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle
from tensorflow.keras.models import load_model
import cv2
from fer import FER

# Load datasets
def load_data():
    # Load the survey.csv dataset
    survey_data = pd.read_csv('survey.csv')

    # Load the reddit_mental_health.csv dataset
    reddit_data = pd.read_csv('reddit_mental_health.csv')

    # Preprocess datasets
    survey_data['text'] = survey_data['comments'].fillna('')
    survey_data['target'] = survey_data['treatment'].apply(lambda x: 1 if x == 'Yes' else 0)
    survey_data['source'] = 'survey'
    
    # For the Reddit dataset, you may need to adjust column names and preprocessing
    reddit_data['source'] = 'reddit_mental_health'
    reddit_data['text'] = reddit_data['content'].fillna('')  # Adjust column names if needed
    reddit_data['target'] = reddit_data['label']  # Ensure this column exists in your dataset

    # Combine datasets
    combined_data = pd.concat([survey_data[['text', 'target', 'source']], reddit_data[['text', 'target', 'source']]])
    return combined_data

# Text preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc])
    return text

# Train and save the model
def train_save_model(data):
    tfidf = TfidfVectorizer(max_features=5000)
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    X = tfidf.fit_transform(data['cleaned_text']).toarray()
    y = data['target']

    # Save the TF-IDF vectorizer
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Save the model
    model.save('mental_health_model.h5')

# Load the saved model and TF-IDF vectorizer
def load_model_and_vectorizer():
    model = load_model('mental_health_model.h5')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

def recommend_intervention(prediction):
    if prediction == 1:
        return "Recommend professional counseling."
    else:
        return "Continue monitoring for any significant changes."

# Function to make predictions
def predict_intervention(text, model, tfidf):
    processed_text = preprocess_text(text)
    features = tfidf.transform([processed_text]).toarray()
    prediction = model.predict(features)
    intervention = recommend_intervention(int(prediction[0][0] > 0.5))
    return prediction[0][0], intervention

# Function to capture live video and analyze emotions
def analyze_video():
    emotion_detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit the video capture.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze emotions in the frame
        result = emotion_detector.detect_emotions(frame)
        if result:
            emotions = result[0]["emotions"]
            predominant_emotion = max(emotions, key=emotions.get)
            text = f"Emotion: {predominant_emotion}"

            # Display the emotion on the frame
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    combined_data = load_data()

    # Train and save the model
    train_save_model(combined_data)

    # Load the model and vectorizer
    model, tfidf = load_model_and_vectorizer()

    # Example usage
    example_text = "I'm feeling really down and hopeless."
    prediction, intervention = predict_intervention(example_text, model, tfidf)
    print(f"Prediction: {prediction}")
    print(f"Intervention: {intervention}")

    # Start video analysis
    analyze_video()
