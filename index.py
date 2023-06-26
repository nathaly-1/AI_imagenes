from flask import Flask, jsonify, send_file, render_template
from keras.models import load_model
from numpy.random import randn
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model("modelo.h5")  # Reemplaza "ruta/a/tu/modelo.h5" con la ruta a tu modelo generado

def generate_image():
    latent_dim = 100
    latent_points = randn(latent_dim)
    x_input = latent_points.reshape(1, latent_dim)
    X = model.predict(x_input)
    array = np.array(X.reshape(100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    return image

@app.route("/")
def serve_index():
    return render_template("index.html")

@app.route("/image")
def serve_image():
    image = generate_image()
    image.save("static/generated_image.jpg")  # Guarda la imagen generada en un archivo
    return render_template("image.html")
    # return jsonify({"message": "Image served successfully!"})

if __name__ == "__main__":
    app.run(debug=True)