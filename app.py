from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import io
import base64

app = Flask(__name__)

def parse_arff_file(filename):
    """
    Parse ARFF file and extract humidity and temperature data
    """
    try:
        # Load ARFF file using scipy
        data, meta = arff.loadarff(filename)
        
        # Convert to numpy arrays
        humidity = np.array([row['humidity'] for row in data])
        temperature = np.array([row['temperature'] for row in data])
        
        return humidity, temperature
    except:
        # Fallback: manual parsing if scipy fails
        humidity = []
        temperature = []
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        data_start = False
        for line in lines:
            line = line.strip()
            if line == '@data':
                data_start = True
                continue
            if data_start and line and not line.startswith('%'):
                parts = line.split(',')
                if len(parts) >= 3:
                    # Format: outlook,temperature,humidity,windy,play
                    temp_val = float(parts[1])
                    humidity_val = float(parts[2])
                    temperature.append(temp_val)
                    humidity.append(humidity_val)
        
        return np.array(humidity), np.array(temperature)

def normal_equation(X, y):
    """
    Implement Linear Regression using Normal Equation
    θ = (XᵀX)⁻¹ Xᵀy
    """
    # Add bias term (column of ones) to X
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Normal equation: θ = (XᵀX)⁻¹ Xᵀy
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    return theta

def cost_function(X, y, theta):
    """
    Implement Cost Function from scratch
    J(θ) = (1/2m) * Σ(hθ(x) - y)²
    """
    m = len(y)
    
    # Add bias term to X
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Calculate predictions
    predictions = X_b @ theta
    
    # Calculate cost
    squared_errors = (predictions - y) ** 2
    cost = (1 / (2 * m)) * np.sum(squared_errors)
    
    return cost, predictions

def predict_temperature(humidity_value, theta):
    """
    Predict temperature using learned parameters
    """
    # Add bias term
    X_b = np.array([1, humidity_value])
    prediction = X_b @ theta
    return prediction

def create_plot_base64(humidity, temperature, theta, user_humidity, user_temp, user_prediction):
    """
    Create visualization and return as base64 string
    """
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(humidity, temperature, marker='o', color='gray', s=50, label='Data points')
    
    # Plot regression line
    humidity_range = np.linspace(humidity.min() - 5, humidity.max() + 5, 100)
    temp_predictions = theta[0] + theta[1] * humidity_range
    plt.plot(humidity_range, temp_predictions, 'b-', linewidth=2, 
             label=f'Temperature = {theta[0]:.2f} + {theta[1]:.2f} * Humidity')
    
    # Plot user's input point
    plt.scatter(user_humidity, user_temp, marker='X', color='red', s=200, 
                label=f'Your input (predicted: {user_prediction:.2f}°F)')
    
    # Chart settings
    plt.title('Linear Regression: Humidity vs Temperature')
    plt.xlabel('Humidity')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

# Load data once at startup
humidity_data, temperature_data = parse_arff_file('weather.numeric.arff')
X_data = humidity_data.reshape(-1, 1)
y_data = temperature_data
theta_data = normal_equation(X_data, y_data)
total_cost_data, _ = cost_function(X_data, y_data, theta_data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        user_humidity = float(request.form['humidity'])
        user_temp = float(request.form['temperature'])
        
        # Make prediction
        user_prediction = predict_temperature(user_humidity, theta_data)
        
        # Calculate cost contribution
        user_cost = (1/2) * ((user_prediction - user_temp) ** 2)
        
        # Calculate error
        error = abs(user_prediction - user_temp)
        
        # Create plot
        plot_base64 = create_plot_base64(humidity_data, temperature_data, 
                                        theta_data, user_humidity, user_temp, user_prediction)
        
        # Prepare results
        results = {
            'equation': f'Temperature = {theta_data[0]:.4f} + {theta_data[1]:.4f} * Humidity',
            'total_cost': f'{total_cost_data:.6f}',
            'user_humidity': user_humidity,
            'user_temp': user_temp,
            'predicted_temp': f'{user_prediction:.2f}',
            'cost_contribution': f'{user_cost:.6f}',
            'absolute_error': f'{error:.2f}',
            'plot_url': f'data:image/png;base64,{plot_base64}',
            'success': True
        }
        
    except ValueError as e:
        results = {
            'error': 'Please enter valid numeric values',
            'success': False
        }
    
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
