# นำเข้าไลบรารีที่จำเป็นสำหรับ Flask Web Application
from flask import Flask, render_template, request  # Flask สำหรับสร้างเว็บแอป
import numpy as np  # ไลบรารีสำหรับการคำนวณทางคณิตศาสตร์
import matplotlib.pyplot as plt  # ไลบรารีสำหรับสร้างกราฟ
from scipy.io import arff  # ไลบรารีสำหรับอ่านไฟล์ ARFF
import io  # ไลบรารีสำหรับจัดการข้อมูลในหน่วยความจำ
import base64  # ไลบรารีสำหรับเข้ารหัสภาพเป็น base64

# สร้าง Flask Application
app = Flask(__name__)

def parse_arff_file(filename):
    """
    ฟังก์ชันสำหรับอ่านไฟล์ ARFF และดึงข้อมูลความชื้นและอุณหภูมิ
    ARFF คือรูปแบบไฟล์ข้อมูลที่ใช้ในการเรียนรู้ของเครื่องจักร
    """
    try:
        # ใช้ scipy ในการโหลดไฟล์ ARFF
        data, meta = arff.loadarff(filename)
        
        # แปลงข้อมูลเป็น numpy arrays
        humidity = np.array([row['humidity'] for row in data])  # ดึงค่าความชื้น
        temperature = np.array([row['temperature'] for row in data])  # ดึงค่าอุณหภูมิ
        
        return humidity, temperature
    except:
        # ถ้า scipy ล้มเหลว จะใช้วิธีการอ่านไฟล์แบบ manual
        humidity = []
        temperature = []
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        data_start = False
        for line in lines:
            line = line.strip()
            if line == '@data':  # หาส่วนที่เริ่มข้อมูลจริงในไฟล์ ARFF
                data_start = True
                continue
            if data_start and line and not line.startswith('%'):  # ข้ามบรรทัดว่างและคอมเมนต์
                parts = line.split(',')
                if len(parts) >= 3:
                    # รูปแบบข้อมูล: outlook,temperature,humidity,windy,play
                    temp_val = float(parts[1])  # ค่าอุณหภูมิอยู่ในคอลัมน์ที่ 2
                    humidity_val = float(parts[2])  # ค่าความชื้นอยู่ในคอลัมน์ที่ 3
                    temperature.append(temp_val)
                    humidity.append(humidity_val)
        
        return np.array(humidity), np.array(temperature)

def normal_equation(X, y):
    """
    สร้างโมเดล Linear Regression โดยใช้ Normal Equation
    สูตร: θ = (XᵀX)⁻¹ Xᵀy
    วิธีนี้คำนวณหาค่าสัมประสิทธิ์ที่เหมาะสมที่สุดโดยตรง
    """
    # เพิ่ม bias term (คอลัมน์ของเลข 1) ใน X เพื่อให้สมการครอบคลุมจุดตัดแกน Y
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # ใช้ Normal Equation: θ = (XᵀX)⁻¹ Xᵀy
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    return theta  # คืนค่าสัมประสิทธิ์ [theta_0, theta_1]

def cost_function(X, y, theta):
    """
    คำนวณ Cost Function (Mean Squared Error) จากหลักการ
    สูตร: J(θ) = (1/2m) * Σ(hθ(x) - y)²
    ใช้วัดความคลาดเคลื่อนระหว่างค่าพยากรณ์และค่าจริง
    """
    m = len(y)  # จำนวนข้อมูลทั้งหมด
    
    # เพิ่ม bias term ใน X
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # คำนวณค่าพยากรณ์จากโมเดล
    predictions = X_b @ theta
    
    # คำนวณค่าความคลาดเคลื่อนกำลังสอง
    squared_errors = (predictions - y) ** 2
    cost = (1 / (2 * m)) * np.sum(squared_errors)
    
    return cost, predictions

def predict_temperature(humidity_value, theta):
    """
    พยากรณ์อุณหภูมิจากค่าความชื้นโดยใช้สัมประสิทธิ์ที่เรียนรู้แล้ว
    สมการ: temperature = theta_0 + theta_1 * humidity
    """
    # เพิ่ม bias term (ค่า intercept)
    X_b = np.array([1, humidity_value])  # [1, humidity] สำหรับคำนวณ theta_0 + theta_1*humidity
    prediction = X_b @ theta  # คำนวณค่าพยากรณ์
    return prediction

def create_plot_base64(humidity, temperature, theta, user_humidity, user_temp, user_prediction):
    """
    สร้างกราฟการวิเคราะห์และแปลงเป็นรูปแบบ base64 สำหรับแสดงในเว็บ
    """
    plt.figure(figsize=(10, 6))
    
    # พล็อตจุดข้อมูลจริง
    plt.scatter(humidity, temperature, marker='o', color='gray', s=50, label='Data points')
    
    # พล็อตเส้น Linear Regression
    humidity_range = np.linspace(humidity.min() - 5, humidity.max() + 5, 100)
    temp_predictions = theta[0] + theta[1] * humidity_range
    plt.plot(humidity_range, temp_predictions, 'b-', linewidth=2, 
             label=f'Temperature = {theta[0]:.2f} + {theta[1]:.2f} * Humidity')
    
    # พล็อตจุดที่ผู้ใช้ป้อนข้อมูล
    plt.scatter(user_humidity, user_temp, marker='X', color='red', s=200, 
                label=f'Your input (predicted: {user_prediction:.2f}°F)')
    
    # ตั้งค่ากราฟ
    plt.title('Linear Regression: Humidity vs Temperature')
    plt.xlabel('Humidity')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # บันทึกกราฟเป็น base64 string เพื่อแสดงในเว็บ
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()  # ปิดกราฟเพื่อปล่อยหน่วยความจำ
    
    return img_base64

# โหลดข้อมูลและสร้างโมเดลครั้งเดียวตอนเริ่มต้นเพื่อเพิ่มประสิทธิภาพ
humidity_data, temperature_data = parse_arff_file('weather.numeric.arff')  # อ่านข้อมูลสภาพอากาศ
X_data = humidity_data.reshape(-1, 1)  # สร้างเมทริกซ์คุณลักษณะ (ความชื้น)
y_data = temperature_data  # เวกเตอร์เป้าหมาย (อุณหภูมิ)
theta_data = normal_equation(X_data, y_data)  # คำนวณสัมประสิทธิ์การถดถอย
total_cost_data, _ = cost_function(X_data, y_data, theta_data)  # คำนวณค่าความคลาดเคลื่อนรวม

@app.route('/')  # Route สำหรับหน้าแรก
def index():
    return render_template('index.html')  # แสดงหน้า HTML หลัก

@app.route('/predict', methods=['POST'])  # Route สำหรับรับข้อมูลจากฟอร์ม
def predict():
    try:
        # รับค่าจากผู้ใช้จากฟอร์ม HTML
        user_humidity = float(request.form['humidity'])  # ค่าความชื้นที่ผู้ใช้ป้อน
        user_temp = float(request.form['temperature'])  # ค่าอุณหภูมิที่ผู้ใช้ป้อน
        
        # พยากรณ์อุณหภูมิจากค่าความชื้นของผู้ใช้
        user_prediction = predict_temperature(user_humidity, theta_data)
        
        # คำนวณค่าความคลาดเคลื่อนของจุดข้อมูลผู้ใช้
        user_cost = (1/2) * ((user_prediction - user_temp) ** 2)
        
        # คำนวณค่าความคลาดเคลื่อนสัมบูรณ์
        error = abs(user_prediction - user_temp)
        
        # สร้างกราฟการวิเคราะห์
        plot_base64 = create_plot_base64(humidity_data, temperature_data, 
                                        theta_data, user_humidity, user_temp, user_prediction)
        
        # เตรียมผลลัพธ์สำหรับส่งกลับไปแสดงในเว็บ
        results = {
            'equation': f'Temperature = {theta_data[0]:.4f} + {theta_data[1]:.4f} * Humidity',  # สมการการถดถอย
            'total_cost': f'{total_cost_data:.6f}',  # ค่าความคลาดเคลื่อนรวม
            'user_humidity': user_humidity,  # ค่าความชื้นที่ผู้ใช้ป้อน
            'user_temp': user_temp,  # ค่าอุณหภูมิที่ผู้ใช้ป้อน
            'predicted_temp': f'{user_prediction:.2f}',  # ค่าอุณหภูมิที่พยากรณ์
            'cost_contribution': f'{user_cost:.6f}',  # ค่าความคลาดเคลื่อนของผู้ใช้
            'absolute_error': f'{error:.2f}',  # ค่าความคลาดเคลื่อนสัมบูรณ์
            'plot_url': f'data:image/png;base64,{plot_base64}',  # URL รูปภาพกราฟ
            'success': True
        }
        
    except ValueError as e:
        # กรณีผู้ใช้ป้อนข้อมูลไม่ถูกต้อง
        results = {
            'error': 'Please enter valid numeric values',  # ข้อความแจ้งเตือน
            'success': False
        }
    
    return render_template('index.html', results=results)  # ส่งผลลัพธ์กลับไปแสดง

if __name__ == '__main__':
    # เริ่มต้น Flask Web Server
    app.run(debug=True, host='0.0.0.0', port=5000)  # รันบนพอร์ต 5000 โหมด debug
