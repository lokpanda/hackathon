from flask import Flask, request, render_template
import pandas as pd
from sklearn.svm import SVC

app = Flask(__name__)

def classify(file, impacts, outcome, inps):
    data = pd.read_csv(file)
    X = data[impacts]
    Y = data[outcome]
    Y = Y.round()
    clf = SVC(kernel='linear')
    clf.fit(X, Y)
    nx = [inps]
    pred = clf.predict(nx)
    return pred

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get values from the form
        hours = int(request.form.get('hours'))
        working = int(request.form.get('working'))
        painful = int(request.form.get('painful'))
        gritty = int(request.form.get('gritty'))
        sensitive = int(request.form.get('sensitive'))
        TV = int(request.form.get('TV'))
        ac = int(request.form.get('ac'))
        humidity = int(request.form.get('humidity'))
        windy = int(request.form.get('windy'))
        driving = int(request.form.get('driving'))
        blurred = int(request.form.get('blurred'))
        poor = int(request.form.get('poor'))
        read = int(request.form.get('reading'))

        p = classify('synthetic_sample_dataset.csv', ["hours", "working", "painful", "gritty", "sensitive", "TV", "ac", "humidity", "windy", "driving", "blurred", "poor", "read"], "outcome", [hours, working, painful, gritty, sensitive, TV, ac, humidity, windy, driving, blurred, poor, read])

        if int(p[0]) == 1:
            pr = "Normal eye condition"
        elif int(p[0]) == 2:
            pr = "Mild dry eye condition"
        elif int(p[0]) == 3:
            pr = "Moderate dry eye condition"
        else:
            pr = "Severe dry eye condition"
        
        return render_template('index.html', prediction=pr)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
