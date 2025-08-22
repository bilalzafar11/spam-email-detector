from flask import Flask, render_template, request, flash
import joblib
import os

# ==========================
# Initialize Flask App
# ==========================
app = Flask(__name__)
app.secret_key = "super_secret_key"  # Needed for flash messages

# ==========================
# Load Trained Model & Vectorizer
# ==========================
model, vectorizer = None, None
MODEL_PATH = "spam_classifier.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Model & Vectorizer loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model/vectorizer: {e}")
else:
    print("⚠️ Model or Vectorizer file not found!")

# ==========================
# Home Route
# ==========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    email_text = None

    if request.method == "POST":
        email_text = request.form.get("email", "").strip()

        if not email_text:
            flash("⚠️ Please enter email text!", "warning")
        elif not model or not vectorizer:
            flash("❌ Model not loaded properly. Please check files.", "danger")
        else:
            try:
                # Transform input text
                email_vector = vectorizer.transform([email_text])

                # Make prediction
                result = model.predict(email_vector)[0]
                proba = model.predict_proba(email_vector)[0]

                # Map prediction to readable format
                prediction = "🚫 Spam" if result == 1 else "✅ Not Spam"
                probability = f"{max(proba) * 100:.2f}% confidence"

            except Exception as e:
                flash(f"❌ Error during prediction: {e}", "danger")

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        email_text=email_text
    )

# ==========================
# Run Flask App (Replit Ready)
# ==========================
if __name__ == "__main__":
    # Replit provides the PORT via environment variables
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True)
