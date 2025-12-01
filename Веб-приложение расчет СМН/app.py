from flask import Flask, render_template, request, jsonify
import numpy as np
import os

app = Flask(__name__)

# Проверка загрузки модели нейронной сети для прогнозирования 
MODEL_PATH = 'model.keras'
model = None

try:
    from tensorflow import keras
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model("C:/Users/User/Desktop/Data Science/FlaskWebProject1/model.keras")
        print("Модель успешно загружена!")
    else:
        print(f"ВНИМАНИЕ: Файл модели '{MODEL_PATH}' не найден. Поместите вашу модель в корень проекта.")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Модель не загружена. Проверьте наличие файла model.keras'}), 500

        # Получаем значения 8 параметров-независимых переменных из JSON
        data = request.get_json()
        inputs = data.get('inputs', [])

        if len(inputs) != 8:
            return jsonify({'error': 'Требуется ровно 8 входных параметров'}), 400

        # Преобразуем в numpy массив
        input_array = np.array([float(x) for x in inputs]).reshape(1, -1)

        # Делаем прогноз-рекомендацию соотношения матрица-наполнитель
        prediction = model.predict(input_array, verbose=0)
        result = float(prediction[0][0])

        return jsonify({
            'success': True,
            'prediction': result,
            'inputs': inputs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
