import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modified_predict_proba_xgb_10.pkl','rb'))


@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    age = int(request.form['age'])
    temperature = request.form['temperature']
    pulse = request.form['pulse']
    rr = request.form['rr']
    rhonchi = request.form['rhonchi']
    wheezes = request.form['wheezes']
    cough = request.form['cough']
    fever = request.form['fever']
    loss_of_smell = request.form['loss_of_smell']
    loss_of_taste = request.form['loss_of_taste']
    listt = [[age, temperature, pulse, rr, rhonchi, wheezes, cough, fever, loss_of_smell, loss_of_taste]];

    prediction = model.predict_proba(np.array(listt, dtype='f'))[:,0]
    pred = model.predict_proba(np.array(listt, dtype='f'))[:,1]

    p = (prediction*100)
    pp = (pred*100)

    return render_template('after.html', output='Probability To Negative: {}% Probability To Positive: {}%'.format(p, pp))

if __name__ == "__main__":
    app.run(debug=True)