from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.graph_objects as go


app = Flask(__name__)

# Load dataset
data = pd.read_csv('heart.csv')


# Preprocess data
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_model = joblib.load('best_model.pkl')

@app.route('/')
def index():
    # Define model performance metrics 
    model_accuracies = {
    "SVM": { 
        "Precision": 72.73,
        "Recall": 82.76,
        "F1-Score": 77.42,
        "Accuracy": 77.05
    },
    "Random Forest": { 
        "Precision": 78.79,
        "Recall": 89.66,
        "F1-Score": 83.87,
        "Accuracy": 83.61
    },
    "Gradient Boosting": { 
        "Precision": 72.73,
        "Recall": 82.76,
        "F1-Score": 77.42,
        "Accuracy": 77.05
    },
    "Naive Bayes": { 
        "Precision": 83.33,
        "Recall": 86.21,
        "F1-Score": 84.75,
        "Accuracy": 85.25
    },
    "Decision Tree": { 
        "Precision": 73.08,
        "Recall": 65.52,
        "F1-Score": 69.09,
        "Accuracy": 72.13
    },
    "Stacking Classifier": { 
        "Precision": 71.43,
        "Recall": 86.21,
        "F1-Score": 78.13,
        "Accuracy": 77.05
    }
}

    # Prepare data for Plotly chart
    model_names = list(model_accuracies.keys())
    metrics = list(model_accuracies[model_names[0]].keys())  # ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    
    # Prepare the traces for each metric
    traces = []
    for i, metric in enumerate(metrics):
        values = [model_accuracies[model][metric] for model in model_names]
        trace = go.Bar(
            x=model_names,
            y=values,
            name=metric,
            text=values, 
            hoverinfo='text',  
        )
        traces.append(trace)

    # Create the layout
    layout = go.Layout(
        title="Model Performance Comparison (Precision, Recall, F1-Score, Accuracy)",
        xaxis=dict(title="Models"),
        yaxis=dict(title="Scores (%)"),
        barmode='group', 
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Convert the plot to HTML div format to be embedded in the template
    plot_div = fig.to_html(full_html=False)

    return render_template('index.html', plot_div=plot_div)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            # Extract user input from the form
            user_input = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['restecg']),
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]

            # Scale user input
            user_input_scaled = scaler.transform([user_input])

            # Make a prediction
            prediction = best_model.predict(user_input_scaled)

            # Prepare the result
            result = 'High risk of heart disease.' if prediction[0] == 1 else 'Low risk of heart disease.'
        except Exception as e:
            result = f"Error in prediction: {str(e)}"

    return render_template('predict.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)