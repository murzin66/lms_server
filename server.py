from flask_socketio import SocketIO

from flask import Flask, Response, jsonify
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
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4343, debug=True)