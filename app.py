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
        try:
            data = CustomData(
                carat=float(request.form["carat"]),
                depth=float(request.form["depth"]),
                table=float(request.form["table"]),
                x=float(request.form["x"]),
                y=float(request.form["y"]),
                z=float(request.form["z"]),
                cut=request.form["cut"],
                color=request.form["color"],
                clarity=request.form["clarity"],
            )

            final_data = data.get_data_as_dataframe()
            predict_pipeline = Predictpipeline()
            prediction = predict_pipeline.predict(final_data)

            final_result = round(prediction[0], 2)

            print(f"Prediction: {final_result}")  # Debugging statement

            return render_template("result.html", final_result=final_result)

        except ValueError as e:
            return f"Error: {str(e)} - Invalid input detected. Please ensure all fields are filled correctly."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
