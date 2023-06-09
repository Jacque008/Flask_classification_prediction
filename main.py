import json
import pandas as pd
from utilities import load_mean_predict
from flask import Flask, request, render_template,jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict',methods=['POST'])
def predict():
        data = request.form['json_data']
        data_list = json.loads(data)  
        df = pd.DataFrame(data_list)
        output = load_mean_predict(df,0.8,0.1,0.1)
        output['Compensable'] = output['Compensable'].apply(lambda x: f'<span style="color:red;">{x}</span>' if x else f'<span style="color:aqua;">{x}</span>')
        output_html = output.to_html(justify='center', escape=False, classes='dataframe')
        return render_template('index.html', table_html=output_html)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json
    df = pd.DataFrame(data)
    output = load_mean_predict(df,0.8,0.1,0.1)
    output = output[['insuranceCaseId','Compensable']]
    result_json = output.to_json(orient='records')
    result_list = json.loads(result_json)
    return json.dumps(result_list)

if __name__ == "__main__":
    app.run(port=5000, debug=False)
    