import joblib
import pandas as pd
from django.shortcuts import render, redirect
from .forms import ContactForm
import joblib
import pandas as pd

# Load the trained model and the label encoder
model = joblib.load('loan_model.joblib')
le = joblib.load('loan_status_label_encoder.joblib')

def home(request):
    """Renders the home page."""
    return render(request, 'home.html')

def contact(request):
    """Renders the contact page and handles form submission."""
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # In a real application, you would handle the form data,
            # like sending an email.
            # For this example, we'll just redirect to the same page
            # with a success message.
            return render(request, 'contact.html', {'form': ContactForm(), 'success': True})
    else:
        form = ContactForm()
    return render(request, 'contact.html', {'form': form})

def predict(request):
    """
    Handles the prediction logic.
    - For GET requests, it displays the prediction form.
    - For POST requests, it processes the form data, makes a prediction,
      and displays the result.
    """
    if request.method == 'POST':
        # Extract data from the form
        age = int(request.POST.get('age'))
        gender = request.POST.get('gender')
        occupation = request.POST.get('occupation')
        education_level = request.POST.get('education_level')
        marital_status = request.POST.get('marital_status')
        income = int(request.POST.get('income'))
        credit_score = int(request.POST.get('credit_score'))

        # Create a DataFrame from the user's input
        # The order of columns must match the order used during model training
        input_data = pd.DataFrame({
            'gender': [gender],
            'occupation': [occupation],
            'education_level': [education_level],
            'marital_status': [marital_status],
            'age': [age],
            'income': [income],
            'credit_score': [credit_score]
        })

        # The model pipeline will automatically handle one-hot encoding
        # for the categorical features.
        prediction_encoded = model.predict(input_data)
        
        # Decode the prediction to a human-readable format
        prediction_label = le.inverse_transform(prediction_encoded)
        
        result = prediction_label[0]
        result_class = 'green' if result == 'Approved' else 'red'

        context = {
            'result': f"The loan is likely to be {result}.",
            'result_class': result_class
        }
        return render(request, 'predict.html', context)

    # If it's a GET request, just render the form
    return render(request, 'predict.html')