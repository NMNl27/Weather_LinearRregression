# นำเข้าไลบรารีที่จำเป็นสำหรับการวิเคราะห์ Linear Regression
import numpy as np  # ไลบรารีสำหรับการคำนวณทางคณิตศาสตร์
import matplotlib.pyplot as plt  # ไลบรารีสำหรับสร้างกราฟ
from scipy.io import arff  # ไลบรารีสำหรับอ่านไฟล์ ARFF
import sys  # ไลบรารีสำหรับจัดการระบบและ input/output

def parse_arff_file(filename):
    """
    ฟังก์ชันสำหรับอ่านไฟล์ ARFF และดึงข้อมูลความชื้นและอุณหภูมิ
    ARFF คือรูปแบบไฟล์ข้อมูลมาตรฐานในการเรียนรู้ของเครื่องจักร
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
    วิธีนี้คำนวณหาค่าสัมประสิทธิ์ที่เหมาะสมที่สุดโดยตรง ไม่ต้องใช้การวนซ้ำ
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
    ค่าที่น้อยกว่าแสดงว่าโมเดลทำงานได้ดีขึ้น
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
    โดย theta_0 คือจุดตัดแกน Y และ theta_1 คือความชันของเส้นตรง
    """
    # เพิ่ม bias term (ค่า intercept)
    X_b = np.array([1, humidity_value])  # [1, humidity] สำหรับคำนวณ theta_0 + theta_1*humidity
    prediction = X_b @ theta  # คำนวณค่าพยากรณ์
    return prediction

def plot_regression(humidity, temperature, theta, user_humidity, user_temp, user_prediction):
    """
    สร้างกราฟการวิเคราะห์ Linear Regression ด้วย matplotlib
    แสดงจุดข้อมูลจริง เส้นการถดถอย และจุดข้อมูลของผู้ใช้
    """
    plt.figure(figsize=(10, 6))
    
    # พล็อตจุดข้อมูลจริงทั้งหมด
    plt.scatter(humidity, temperature, marker='o', color='gray', s=50, label='Data points')
    
    # พล็อตเส้น Linear Regression ที่คำนวณได้
    humidity_range = np.linspace(humidity.min() - 5, humidity.max() + 5, 100)
    temp_predictions = theta[0] + theta[1] * humidity_range
    plt.plot(humidity_range, temp_predictions, 'b-', linewidth=2, 
             label=f'Temperature = {theta[0]:.2f} + {theta[1]:.2f} * Humidity')
    
    # พล็อตจุดข้อมูลที่ผู้ใช้ป้อนเข้ามา
    plt.scatter(user_humidity, user_temp, marker='X', color='red', s=200, 
                label=f'Your input (predicted: {user_prediction:.2f}°F)')
    
    # ตั้งค่ากราฟ
    plt.title('Linear Regression: Humidity vs Temperature')
    plt.xlabel('Humidity')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()  # จัดรูปแบบกราฟให้พอดี
    plt.show()  # แสดงกราฟ

def main():
    """
    ฟังก์ชันหลักสำหรับรันการวิเคราะห์ Linear Regression ทั้งหมด
    แสดงขั้นตอนการทำงานตั้งแต่การโหลดข้อมูลจนถึงการแสดงผลลัพธ์
    """
    print("=" * 60)
    print("LINEAR REGRESSION: HUMIDITY VS TEMPERATURE ANALYSIS")
    print("วิเคราะห์ความสัมพันธ์ระหว่างความชื้นและอุณหภูมิด้วย Linear Regression")
    print("=" * 60)
    
    # อ่านไฟล์ ARFF เพื่อดึงข้อมูล
    filename = 'weather.numeric.arff'
    try:
        humidity, temperature = parse_arff_file(filename)
        print(f"✓ โหลดข้อมูล {len(humidity)} จุด จากไฟล์ {filename} สำเร็จ")
    except Exception as e:
        print(f"✗ เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
        return
    
    print(f"✓ ช่วงค่าความชื้น: {humidity.min():.1f} - {humidity.max():.1f}")
    print(f"✓ ช่วงค่าอุณหภูมิ: {temperature.min():.1f} - {temperature.max():.1f}°F")
    print()
    
    # สร้างโมเดล Linear Regression โดยใช้ Normal Equation
    X = humidity.reshape(-1, 1)  # เมทริกซ์คุณลักษณะ (ความชื้น)
    y = temperature              # เวกเตอร์เป้าหมาย (อุณหภูมิ)
    
    theta = normal_equation(X, y)
    print(f"✓ สมการการถดถอยที่ได้:")
    print(f"  Temperature = {theta[0]:.4f} + {theta[1]:.4f} * Humidity")
    print()
    
    # คำนวณค่าความคลาดเคลื่อนรวมสำหรับข้อมูลทั้งหมด
    total_cost, all_predictions = cost_function(X, y, theta)
    print(f"✓ ค่าความคลาดเคลื่อนรวม J(θ) สำหรับข้อมูล {len(humidity)} จุด: {total_cost:.6f}")
    print()
    
    # รับข้อมูลจากผู้ใช้
    print("ข้อมูลจากผู้ใช้:")
    print("-" * 30)
    # ตรวจสอบว่ากำลังรันในโหมด interactive หรือไม่
    if sys.stdin.isatty():
        try:
            user_humidity = float(input("ป้อนค่าความชื้น: "))
            user_temp = float(input("ป้อนค่าอุณหภูมิ (°F): "))
        except ValueError:
            print("✗ ข้อมูลไม่ถูกต้อง กรุณาป้อนเฉพาะตัวเลข")
            return
    else:
        # ใช้ค่าตัวอย่างสำหรับโหมด non-interactive
        user_humidity = 85.0
        user_temp = 75.0
        print(f"ใช้ค่าตัวอย่าง: ความชื้น = {user_humidity}, อุณหภูมิ = {user_temp}")
    
    # พยากรณ์อุณหภูมิจากค่าความชื้นของผู้ใช้
    user_prediction = predict_temperature(user_humidity, theta)
    print()
    print("ผลลัพธ์:")
    print("-" * 30)
    print(f"✓ อุณหภูมิที่พยากรณ์จากความชื้น ({user_humidity}): {user_prediction:.2f}°F")
    
    # คำนวณค่าความคลาดเคลื่อนของจุดข้อมูลผู้ใช้
    user_cost = (1/2) * ((user_prediction - user_temp) ** 2)
    print(f"✓ ค่าความคลาดเคลื่อนของข้อมูลผู้ใช้: {user_cost:.6f}")
    
    # คำนวณและแสดงค่าความคลาดเคลื่อนสัมบูรณ์
    error = abs(user_prediction - user_temp)
    print(f"✓ ค่าความคลาดเคลื่อนสัมบูรณ์: {error:.2f}°F")
    print()
    
    # สร้างกราฟการวิเคราะห์
    print("กำลังสร้างกราฟการวิเคราะห์...")
    plot_regression(humidity, temperature, theta, user_humidity, user_temp, user_prediction)
    
    print("=" * 60)
    print("การวิเคราะห์เสร็จสิ้น!")
    print("=" * 60)

if __name__ == "__main__":
    main()
