from flask import render_template, request, jsonify
from app import app, db
import joblib
import numpy as np
import os

MODEL_SVM = os.path.join(os.path.dirname(__file__), 'data', 'model', 'svm_model_and_vectorizer.pkl')

# Memuat model dan vectorizer
model, vectorizer = joblib.load(MODEL_SVM)


# Memuat model dan vectorizer yang telah disimpan
# with open(MODEL_SVM, 'rb') as f:
#     model, vectorizer = pickle.load(f)

@app.route('/')
def home():
    conn = db.get_db()
    cursor = conn.cursor()

    # Query 1: Total number of analyzed titles
    cursor.execute("SELECT COUNT(*) FROM history_pred")
    total_titles = cursor.fetchone()[0]

    # Query 2: Sentiment distribution (Positif, Negatif, Netral)
    cursor.execute("SELECT hasil, COUNT(*) as count FROM history_pred GROUP BY hasil")
    sentiment_data = cursor.fetchall()
    
    # Initialize sentiment counts
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    # Process the results
    for row in sentiment_data:
        if row['hasil'] == 'Positif':
            negative_count = row['count']
        elif row['hasil'] == 'Negatif':
            positive_count = row['count']
        elif row['hasil'] == 'Netral':
            neutral_count = row['count']
    
    # Calculate sentiment percentages
    total_sentiment = positive_count + negative_count + neutral_count
    positive_percentage = (positive_count / total_sentiment) * 100 if total_sentiment else 0
    negative_percentage = (negative_count / total_sentiment) * 100 if total_sentiment else 0
    neutral_percentage = (neutral_count / total_sentiment) * 100 if total_sentiment else 0

    # Query 3: Latest 3 analyses for the table
    cursor.execute("SELECT judul_berita, hasil, ROUND(score, 2) AS score FROM history_pred ORDER BY create_at DESC")
    recent_analyses = cursor.fetchall()

    # Convert sqlite3.Row objects to dictionaries for easier access in Jinja2
    recent_analyses = [dict(row) for row in recent_analyses]

    conn.close()

    # Prepare data to pass to the template
    data = {
        'total_titles': total_titles,
        'positive_percentage': "{:.2f}".format(positive_percentage),
        'negative_percentage': "{:.2f}".format(negative_percentage),
        'neutral_percentage': "{:.2f}".format(neutral_percentage),
        'recent_analyses': recent_analyses
    }

    return render_template('home.html', data=data)


@app.route('/analisa')
def analisa():
    return render_template('analisa.html')

@app.route('/analyze', methods=["POST"])
def analyze_sentiment():
    # Mendapatkan data dari request
    data = request.get_json()
    title = data.get('title', '')  # Mengambil 'title' dari request

    if not title:
        return jsonify({'error': 'Judul berita diperlukan'}), 400

    # Mengubah teks menjadi vektor
    text_vectorized = vectorizer.transform([title])

    # Mengonversi sparse matrix menjadi dense matrix
    text_vectorized_dense = text_vectorized.toarray()

    # Melakukan prediksi sentimen
    sentiment = model.predict(text_vectorized_dense)
    probability = model.predict_proba(text_vectorized_dense)

    # Menentukan label sentimen berdasarkan hasil prediksi
    if sentiment[0] == 1:
        sentiment_label = 'Positif'
    elif sentiment[0] == 0:
        sentiment_label = 'Negatif'
    elif sentiment[0] == 2:
        sentiment_label = 'Netral'
    else:
        sentiment_label = 'Tidak Dikenal'  # Untuk menangani kasus yang tak terduga

    # Save result to the database
    db.insert_into_history(title, sentiment_label, float(max(probability[0])))

    # Menyiapkan hasil prediksi dan probabilitas
    result = {
        'title': title,  # Mengirimkan kembali judul yang dianalisis
        'sentiment': sentiment_label,  # Mengirimkan label sentimen yang lebih deskriptif
        'score': float(max(probability[0])),  # Konversi ke tipe float
    }

    return jsonify(result)