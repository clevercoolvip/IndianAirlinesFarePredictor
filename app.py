from flask import *
import keras
import numpy as np

model = keras.models.load_model("models/model_Opt_RMSprop/")

app = Flask(__name__)
global input_class_names
input_class_names = [['Air India',
                      'GoAir',
                      'IndiGo',
                      'SpiceJet',
                      'Vistara',
                      'other'], ['Chennai',
                                 'Delhi',
                                 'Kolkata',
                                 'Mumbai'], ['Cochin',
                                             'Delhi',
                                             'Hyderabad',
                                             'Kolkata'],
                     ['evening',
                      'morning'],
                     ['Date_of_Journey']]


def pred(lst):
    from itertools import chain
    arr = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0], [0]]
    for j in range(len(input_class_names)):
        if lst[j] in input_class_names[j]:
            arr[j][input_class_names[j].index(lst[j])] = 1

        if lst[j] == "weekday":
            arr[len(arr) - 1][0] = 1
        else:
            arr[len(arr) - 1][0] = 0

    return list(chain.from_iterable(arr))


@app.route("/")
def alive():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def getting_values():
    if request.method == "POST":
        company = request.form['airplane_wala']
        source = request.form['source_wala']
        destination = request.form['destination_wala']
        time_ = request.form['time_wala']
        day_ = request.form['day_wala']
        lst_values = [company, source, destination, time_, day_]
        inp = pred(lst_values)
        out = np.round(model.predict(np.expand_dims(inp, axis=0)))[0][0]
        return render_template("index.html", prediction=out)

    else:
        return redirect(url_for("alive"))


if __name__ == '__main__':
    app.run(debug=True, port=8000)
