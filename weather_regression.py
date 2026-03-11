import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import sys

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

def plot_regression(humidity, temperature, theta, user_humidity, user_temp, user_prediction):
    """
    Create visualization with matplotlib
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
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the linear regression analysis
    """
    print("=" * 60)
    print("LINEAR REGRESSION: HUMIDITY VS TEMPERATURE ANALYSIS")
    print("=" * 60)
    
    # Parse ARFF file
    filename = 'weather.numeric.arff'
    try:
        humidity, temperature = parse_arff_file(filename)
        print(f"✓ Successfully loaded {len(humidity)} data points from {filename}")
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return
    
    print(f"✓ Humidity range: {humidity.min():.1f} - {humidity.max():.1f}")
    print(f"✓ Temperature range: {temperature.min():.1f} - {temperature.max():.1f}°F")
    print()
    
    # Implement Linear Regression using Normal Equation
    X = humidity.reshape(-1, 1)  # Feature matrix
    y = temperature              # Target vector
    
    theta = normal_equation(X, y)
    print(f"✓ Learned regression equation:")
    print(f"  Temperature = {theta[0]:.4f} + {theta[1]:.4f} * Humidity")
    print()
    
    # Calculate total cost for all data points
    total_cost, all_predictions = cost_function(X, y, theta)
    print(f"✓ Total Cost J(θ) for all {len(humidity)} data points: {total_cost:.6f}")
    print()
    
    # Get user input
    print("USER INPUT:")
    print("-" * 30)
    # Check if running in interactive mode
    if sys.stdin.isatty():
        try:
            user_humidity = float(input("Enter humidity value: "))
            user_temp = float(input("Enter temperature (°F) value: "))
        except ValueError:
            print("✗ Invalid input. Please enter numeric values.")
            return
    else:
        # Use demo values for non-interactive mode
        user_humidity = 85.0
        user_temp = 75.0
        print(f"Using demo values: humidity = {user_humidity}, temperature = {user_temp}")
    
    # Predict temperature for user's humidity
    user_prediction = predict_temperature(user_humidity, theta)
    print()
    print("RESULTS:")
    print("-" * 30)
    print(f"✓ Predicted temperature from your humidity ({user_humidity}): {user_prediction:.2f}°F")
    
    # Calculate cost contribution of user's input point
    user_cost = (1/2) * ((user_prediction - user_temp) ** 2)
    print(f"✓ Cost contribution of your input point: {user_cost:.6f}")
    
    # Calculate and display error
    error = abs(user_prediction - user_temp)
    print(f"✓ Absolute error: {error:.2f}°F")
    print()
    
    # Create visualization
    print("Generating visualization...")
    plot_regression(humidity, temperature, theta, user_humidity, user_temp, user_prediction)
    
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
