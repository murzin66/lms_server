import requests
from flask_socketio import SocketIO
import hashlib
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import json
import os
import torch
import torch.nn.functional as F
import joblib
from torch_geometric.nn import GATConv

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    print('Client connected')
@app.route("/")
def get_courses():
    file_path = os.path.join(os.path.dirname(__file__), 'mockCourseList.json')

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()

        return Response(
            data,
            status=200,
            mimetype='application/json; charset=utf-8'
        )

    except FileNotFoundError:
        return Response(
            json.dumps({"error": "Файл не найден"}, ensure_ascii=False),
            status=404,
            mimetype='application/json; charset=utf-8'
        )
    except json.JSONDecodeError as e:
        return Response(
            json.dumps({"error": f"Ошибка формата JSON: {str(e)}"}, ensure_ascii=False),
            status=500,
            mimetype='application/json; charset=utf-8'
        )
    except Exception as e:
        return Response(
            json.dumps({"error": f"Непредвиденная ошибка: {str(e)}"}, ensure_ascii=False),
            status=500,
            mimetype='application/json; charset=utf-8'
        )


def get_course(course_Id):
    file_path = os.path.join(os.path.dirname(__file__), 'mockCourses.json')

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            courses = data.get('courses', [])

            for course in courses:
                if isinstance(course, dict) and course.get('courseId') == course_Id:
                    return course
            return None

    except FileNotFoundError:
        return {"error": "Файл с курсами не найден"}, 500
    except json.JSONDecodeError:
        return {"error": "Ошибка формата данных курсов"}, 500


@app.route('/course/<int:course_Id>', methods=['GET'])
def course_api(course_Id):
    result = get_course(course_Id)

    if result:
        return Response(
            json.dumps(result),
            status=200,
            mimetype='application/json; charset=utf-8'
        )
    else:
        return jsonify({
            "status": "error",
            "message": f"Курс с ID {course_Id} не найден"
        }), 404

def get_progress(userId):
    file_path = os.path.join(os.path.dirname(__file__), 'mockProgress.json')
    result = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for prog in data:
                if isinstance(prog, dict) and prog.get('userId') == userId:
                    return prog.get('progress')
                return None


    except FileNotFoundError:
        return {"error": "Файл с курсами не найден"}, 500
    except json.JSONDecodeError:
        return {"error": "Ошибка формата данных курсов"}, 500


@app.route('/progress/<int:userId>', methods = ['GET'])
def progress_api(userId):
    result = get_progress(userId)
    if result:
        return Response(
            json.dumps(result),
            status = 200,
            mimetype="application/json; charset = utf-8"
        )
    else:
        return jsonify({
            "status": "error",
            "message": f"Прогресс для пользователя с ID {userId} не найден"
        }), 404


def get_userInfo(userEmail):
    file_path = os.path.join(os.path.dirname(__file__), 'mockUsers.json')
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file).get('users')
            for user in data:
                if isinstance(user, dict) and user.get('email') == userEmail:
                    #print (user)
                    return user
                return {"error": "Пользователь не найден"}, 404


    except FileNotFoundError:
        return {"error": "Файл с курсами не найден"}, 500
    except json.JSONDecodeError:
        return {"error": "Ошибка формата данных курсов"}, 500


@app.route('/user/<string:userEmail>', methods=['GET'])
def userInfo_api(userEmail):
    result = get_userInfo(userEmail)
    print (result)
    if result:
        return Response(
            json.dumps(result),
            status=200,
            mimetype="application/json; charset = utf-8"
        )
    else:
        return jsonify({
            "status": "error",
            "message": f"Информация о пользователе с email: {userEmail} не найдена"
        }), 404

def getSearchResults(query):
    file_path = os.path.join(os.path.dirname(__file__), 'mockSearchResults.json')
    result = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    except FileNotFoundError:
        return {"error": "Файл с курсами не найден"}, 500
    except json.JSONDecodeError:
        return {"error": "Ошибка формата данных курсов"}, 500

@app.route('/search/<string:query>', methods=['GET'])
def search_api(query):
    result = getSearchResults(query)
    if result:
        return Response(
            json.dumps(result),
            status=200,
            mimetype="application/json; charset = utf-8"
        )
    else:
        return jsonify({
            "status": "error",
            "message": f"Информация о {query} не найдена"
        }), 404

def searchForToken(token):
    file_path = os.path.join(os.path.dirname(__file__), 'mockUsers.json')
    result = []
    m = hashlib.sha256()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file).get('users')
            for user in data:
                encoded_text = user.get('email').encode("utf-8")
                m.update(encoded_text)
                if isinstance(user, dict) and m.hexdigest() == token:
                    return user
                return None


    except FileNotFoundError:
        return {"error": "Файл с курсами не найден"}, 500
    except json.JSONDecodeError:
        return {"error": "Ошибка формата данных курсов"}, 500

@app.route('/auth/<string:token>', methods = ['GET'])
def checkAuth_api (token):
    result = searchForToken(token)
    if result:
        return Response(
            json.dumps(result),
            status=200,
            mimetype="application/json; charset = utf-8"
        )
    else:
        return jsonify({
            "status": "error",
            "message": f"Информация о пользователе {token} не найдена"
        }), 404

def checkLogin(email, password):
    m = hashlib.sha256()
    encoded_text = email.encode("utf-8")
    m.update(encoded_text)
    user = searchForToken(m.hexdigest())
    if user.get('password') == password:
        return jsonify({
            "email": email,
            "token": m.hexdigest(),
            "id": user.get("userId")
        })
    else:
        return jsonify({
            "status": "error",
            "message": f"Информация о полдьзователе {email} не найдена"
        }), 404

@app.route('/login', methods = ['POST'])
def login_api():
    email = request.form.get('email')
    password = request.form.get('password')
    result = checkLogin(email, password)
    if result:
        return result
    else:
        return jsonify({
            "status": "error",
            "message": f"Информация о полдьзователе {email} не найдена"
        }), 404


@app.route('/logout', methods=['DELETE'])
def logout_api():
    token = request.headers.get('X-Token')

    result = searchForToken(token)

    if result:
        return Response(
            json.dumps(result),
            status=200,
            mimetype="application/json; charset=utf-8"
        )
    else:
        return jsonify({
            "status": "error",
            "message": "Недействительный токен"
        }), 401


def changeUserInfo(name, surname, middlename, interests, email, photoUrl):
    with open('mockUsers.json', 'r+', encoding='utf-8') as file:
        data = json.load(file)
        updated_user = None

        for user in data['users']:
            if user['email'] == email:
                user.update({
                    'name': name,
                    'surname': surname,
                    'middlename': middlename,
                    'interests': interests,
                    'photoUrl': photoUrl
                })
                updated_user = user.copy()
                updated_user.pop('password', None)
                file.seek(0)
                json.dump(data, file, indent=4, ensure_ascii=False)
                file.truncate()
                break

        if updated_user:
            return json.dumps(updated_user, ensure_ascii=False)

        return json.dumps({'error': 'User not found'}, ensure_ascii=False)
@app.route('/user', methods = ['POST'])
def changeUserInfoApi():
    data = request.get_json()

    name = data.get('name')
    surname = data.get('surname')
    middlename = data.get('middlename')
    interests = data.get('interests')
    email = data.get('email')
    photoUrl = data.get('photoUrl')

    result = changeUserInfo(name, surname, middlename, interests, email, photoUrl)
    if result:
        return Response(
            result,
            status=200,
            mimetype="application/json; charset=utf-8"
        )
    else:
        return jsonify({
            "status": "error",
            "message": "Ошибка обновления данных"
        }), 400


def changeEnrolledCourses(email, courseId):
    with open('mockUsers.json', 'r+', encoding='utf-8') as file:
        data = json.load(file)
        updated_user = None

        for user in data['users']:
            if user['email'] == email:
                enrolled_courses = user.get('enrolledCourses', [])

                if courseId not in enrolled_courses:
                    enrolled_courses.append(courseId)

                user['enrolledCourses'] = enrolled_courses

                updated_user = user.copy()
                updated_user.pop('password', None)

                file.seek(0)
                json.dump(data, file, indent=4, ensure_ascii=False)
                file.truncate()
                break

        if updated_user:
            return json.dumps(updated_user, ensure_ascii=False)

        return json.dumps({'error': 'User not found'}, ensure_ascii=False)
@app.route('/course', methods = ['POST'])
def changeEnrolledCourses_api():
    data = request.get_json()
    email = data.get('email')
    courseId = data.get('courseId')
    result = changeEnrolledCourses(email, courseId)
    print (result)
    if result:
        return Response(
            result,
            status=200,
            mimetype="application/json; charset=utf-8"
        )
    else:
        return jsonify({
            "status": "error",
            "message": "Ошибка изменения списка курсов пользователя"
        }), 400



# Разворачивание обученной модели графовой нейросети
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Загрузка векторайзера
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Определение устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализация модели
model = GAT(in_channels=len(vectorizer.get_feature_names_out()), hidden_channels=8, out_channels=8, heads=8, dropout=0.5).to(device)

# Загрузка весов модели
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()

# Словарь для отображения индексов классов в названия
class_mapping = {
    0: 'DevOps',
    1: 'Web',
    2: 'ML',
    3: 'Software',
    4: 'Art',
    5: 'Biology',
    6: 'Physics',
    7: 'Math'
}


def predict(description):

    if not description:
        return jsonify({'error': 'Описание предпочтений не предоставлено'}), 400

    # Векторизация нового описания
    x_new = vectorizer.transform([description]).toarray()
    x_new = torch.tensor(x_new, dtype=torch.float).to(device)

    # Создание пустого edge_index, так как у нас нет связей для нового узла
    edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

    # Предсказание класса
    with torch.no_grad():
        out = model(x_new, edge_index)
        pred = out.argmax(dim=1).item()
        predicted_class = class_mapping.get(pred, 'Неизвестный класс')

    return predicted_class

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4343, debug=True)