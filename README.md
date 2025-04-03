# Air-Quality-Prediction

This project predicts air quality using machine learning models (Linear Regression and Decision Tree Regression).  
The model is trained on an air quality dataset and allows users to input values through a web interface to get predictions.

📂Project Structure
 ├── app.py              # Flask web server
 ├── train.py            # Model training script
 ├── air_quality_model.pkl # Trained model file (generated after training)
 ├── templates/
 │   ├── index.html      # Frontend UI
 ├── static/
 │   ├── style.css       # Stylesheet for UI
 ├── air_quality.csv     # Dataset 
 ├── requirements.txt    # Required dependencies

 
## Installation & Setup

### 
1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/AirQualityPrediction.git
cd AirQualityPrediction
```
2️⃣ Create and Activate a Virtual Environment
```
For Windows:
python -m venv venv
venv\Scripts\activate

For macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

4️⃣Train the Model (Optional)
```
python train.py
```

5️⃣Run the Flask App
```
python app.py
```
