from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import Predictpipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else:
        data = CustomData(
            carat=float(request.form.get("carat", 0.0)),
            depth=float(request.form.get("depth", 0.0)),
            table=float(request.form.get("table", 0.0)),
            x=float(request.form.get("x", 0.0)),
            y=float(request.form.get("y", 0.0)),
            z=float(request.form.get("z", 0.0)),
            cut=request.form.get("cut", ""),
            color=request.form.get("color", ""),
            clarity=request.form.get("clarity", ""),
        )

        final_data = data.get_data_as_dataframe()
        predict_pipeline = Predictpipeline()
        prediction = predict_pipeline.predict(final_data)

        final_result = round(prediction[0], 2)

        print(f"Prediction: {final_result}")  # Debugging statement

        return render_template("result.html", final_result=final_result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
