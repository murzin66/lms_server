from flask import Flask, Response
import json
import os

app = Flask(__name__)


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
        return json.dumps({"error": "Файл не найден"}, ensure_ascii=False), 404
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4343, debug=True)