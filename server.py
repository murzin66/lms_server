from flask_socketio import SocketIO

from flask import Flask, Response
from flask_cors import CORS
import json
import os

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4343, debug=True)