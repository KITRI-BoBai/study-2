import csv
from flask import Flask, render_template, request, redirect, url_for
from flask_restx import Api, Resource
from werkzeug.utils import secure_filename

# Flask 객체에 API 객체 등록
app = Flask(__name__)
api = Api(app)

# count: index를 나타내기 위함
# petition: 카테고리를 나타내기 위함 
count = -1
petition = -1

# 읽기: CSV 파일을 Dictionary로 변경
mydict = {}
with open('./train-result.csv', 'r', encoding='utf-8') as input:
    reader = csv.reader(input)
    mydict = {rows[0]:rows[1] for rows in reader}
    count = len(mydict.keys()) - 1

# /upload: HTML 렌더링
@app.route('/upload')
def render_html():
    return render_template('server.html')

# /input: File 업로드
@app.route('/input', methods = ['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return redirect(url_for('add_result'))

# /add: CSV 파일에 새로운 Input 분석 결과 추가
@app.route('/add')
def add_result():
    global count
    global petition
    with open('./input-result.csv', 'r', encoding='utf-8') as input:
        first_line_flag = False
        reader = csv.reader(input)
        for rows in reader:
            if not first_line_flag:
                first_line_flag = True
                continue
            number = rows[0]
            petition = rows[1]
        f = open('./train-result.csv', 'a', newline='')
        wr = csv.writer(f)
        wr.writerow([count, petition])
        mydict[str(count)] = petition
        count += 1
    return redirect(url_for('print_result'))

# /output: 분석 결과 출력
@app.route('/output')
def print_result():
    return {
        "index": "%s" % str(count - 1),
        "category": "%s" % mydict[str(count - 1)]
    }

# /api/{index}: 학습된 API 가져오기
@api.route('/api/<string:index>')
class API(Resource):
    def get(self, index):
        return {
            "index": "%s" % index,
            "category": "%s" % mydict[index]
        }

# main: localhost 80 port에서 서버 구동
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)