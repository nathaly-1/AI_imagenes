from flask import Flask, jsonify, send_file, redirect
from keras.models import load_model
from numpy.random import randn
from PIL import Image
import numpy as np
# from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam

app = Flask(__name__)
model = load_model("modeloGan.h5")  # Reemplaza "ruta/a/tu/modelo.h5" con la ruta a tu modelo generado
# optimizer = Adam(learning_rate=0.001)  # Especifica el optimizador y su tasa de aprendizaje
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Compilar el modelo
# optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
model.compile(optimizer='adam', loss='binary_crossentropy')

def generate_image():
    latent_dim = 100
    latent_points = randn(latent_dim)
    x_input = latent_points.reshape(1, latent_dim)
    X = model.predict(x_input)
    X_resized = np.array(Image.fromarray(X[0]).resize((100, 100)))

    # Verificar si el rango de valores es válido
    if X_resized.max() - X_resized.min() == 0:
        return None

    # Ajustar la escala de valores de los píxeles
    X_scaled = ((X_resized - X_resized.min()) / (X_resized.max() - X_resized.min())) * 255
    X_scaled = np.nan_to_num(X_scaled)

    array = np.array(X_scaled, dtype=np.uint8)
    image = Image.fromarray(array)
    return image

@app.route("/")
def serve_image():
    image = generate_image()
    if image is None:
        return jsonify({"message": "Invalid image range"})
    # image = generate_image()
    image.save("generated_image.jpg")  # Guarda la imagen generada en un archivo
    # return jsonify({"message": "Image served successfully!"})
    return redirect("/image")

@app.route("/image")
def show_image():
    return send_file("generated_image.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)