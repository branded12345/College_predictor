from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np

application = Flask(__name__)
app = application

scaler = pickle.load(open("Model/standardScaler.pkl", 'rb'))
model = pickle.load(open("Model/modelForPrediction.pkl", 'rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        GRE = float(request.form.get('GRE'))
        TOEFL = float(request.form.get('TOEFL'))
        univ_rat = float(request.form.get('univ_rat'))
        SOP = float(request.form.get('SOP'))
        LOR = float(request.form.get('LOR'))
        CGPA = float(request.form.get('CGPA'))  # Corrected typo here
        res_exp = float(request.form.get('res_exp'))

        new_data = scaler.transform([[GRE, TOEFL, univ_rat, SOP, LOR, CGPA, res_exp]])
        predict = model.predict(new_data)

        if predict[0] == 1:
            result = "You have very high possibility of getting admitted to this college."
        else:
            result = "You have very less possibility of getting admitted to this college."

        return render_template('result.html', result=result)

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
