# app.py
from flask import Flask, render_template, request, send_file
import os
from fatigue_predict import predict_single, predict_file

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prob = None
    error = None
    table = None

    if request.method == "POST":

        # ---------- Single Sample Prediction ----------
        features_str = request.form.get("features")
        if features_str:
            result, prob = predict_single(features_str)
            if "Error" in str(result):
                error = result
                result = None

        # ---------- File Upload Prediction ----------
        file = request.files.get("file")
        if file and file.filename != "":
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            df, err = predict_file(file_path)
            if err:
                error = err
            else:
                table = df.to_html(classes="table table-bordered", index=False)
                result = None  # hide single prediction

    return render_template(
        "index.html",
        result=result,
        prob=prob,
        error=error,
        table=table
    )


if __name__ == "__main__":
    app.run(debug=True)
