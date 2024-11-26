# from flask import Flask, render_template, request, redirect, url_for, send_file
# import os
# import numpy as np
# import pandas as pd
# from keras.models import Sequential, load_model
# from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
# from keras.optimizers import Adam
# from PIL import Image
# import tensorflow as tf

# app = Flask(__name__)

# ROWS = 190
# COLS = 160
# CHANNELS = 3
# TEST_DIR = r'E:\downloads\Signature-recognition-master\Signature-recognition-master\test'
# SIGNATURE_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'K', 'L', 'M', 'N', 'O', 'P']

# # Define model (Use pre-trained model if exists)
# model = Sequential()
# model.add(Activation(activation=lambda x: (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x), input_shape=(ROWS, COLS, CHANNELS)))
# model.add(Convolution2D(64, 3, 3, padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Convolution2D(96, 3, 3, padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Convolution2D(128, 2, 2, padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(SIGNATURE_CLASSES)))
# model.add(Activation('sigmoid'))

# # Compile the model
# adam = Adam(learning_rate=0.0001)
# model.compile(optimizer=adam, loss='categorical_crossentropy')

# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle file upload
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file:
#         filename = file.filename
#         file_path = os.path.join('uploads', filename)
#         file.save(file_path)

#         # Process the uploaded image and make a prediction
#         image = Image.open(file_path)
#         image = image.resize((COLS, ROWS))
#         if image.mode != 'RGB':
#             image = image.convert('RGB')

#         # Convert to numpy array and normalize
#         image_array = np.array(image, dtype=np.float32)  # Cast to float32
#         image_array = image_array.reshape(1, ROWS, COLS, CHANNELS)

#         # Normalize the image using mean and standard deviation
#         def center_normalize(x):
#             return (x - np.mean(x)) / np.std(x)

#         image_array = center_normalize(image_array)

#         # Make the prediction
#         prediction = model.predict(image_array)
#         predicted_class = SIGNATURE_CLASSES[np.argmax(prediction)]

#         # Save the result to the result.html page
#         return render_template('result.html', result=predicted_class)


# if __name__ == '__main__':
#     app.run(debug=True)






















from flask import Flask, render_template, request, redirect
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import Adam
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Constants
ROWS = 190
COLS = 160
CHANNELS = 3
EMBEDDING_DIR = r'E:\downloads\Signature-recognition-master\Signature-recognition-master\embeddings'
SIGNATURES_DIR = r'E:\downloads\Signature-recognition-master\Signature-recognition-master\signatures'

# Build the Siamese network model for signature verification
def build_siamese_model(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(96, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (2, 2), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    embedding = Dense(128, activation=None)(x)
    normalized_embedding = Lambda(lambda y: tf.linalg.l2_normalize(y, axis=-1), output_shape=(128,))(embedding)

    return Model(input_img, normalized_embedding)

# Extract embedding from an image
def extract_embedding(image):
    image = image.resize((COLS, ROWS))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array.reshape(1, ROWS, COLS, CHANNELS)
    image_array = (image_array - np.mean(image_array)) / np.std(image_array)
    embedding = embedding_model.predict(image_array)
    print(f"Extracted embedding shape: {embedding.shape}")
    return embedding

# Save embeddings for predefined signatures
def save_embeddings_from_signatures():
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)
        
    for signature_file in os.listdir(SIGNATURES_DIR):
        if signature_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(SIGNATURES_DIR, signature_file)
            image = Image.open(image_path)
            embedding = extract_embedding(image)

            name = signature_file.split('.')[0]
            reference_embedding_path = os.path.join(EMBEDDING_DIR, f"{name}.npy")
            np.save(reference_embedding_path, embedding)
            print(f"Saved embedding for: {name}")

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        image = Image.open(file)
        uploaded_embedding = extract_embedding(image)

        found_match = False
        for signature_file in os.listdir(SIGNATURES_DIR):
            reference_embedding_path = os.path.join(EMBEDDING_DIR, f"{signature_file.split('.')[0]}.npy")
            if os.path.exists(reference_embedding_path):
                reference_embedding = np.load(reference_embedding_path)
                similarity = np.linalg.norm(uploaded_embedding - reference_embedding[0])
                print(f"Similarity with {signature_file.split('.')[0]}: {similarity}")

                threshold = 0.5
                if similarity < threshold:
                    name = signature_file.split('.')[0]
                    found_match = True
                    return render_template('result.html', result=f"Signature is authentic: {name}")

        if not found_match:
            return render_template('result.html', result="Signature does not match any known signatures.")

if __name__ == '__main__':
    input_shape = (ROWS, COLS, CHANNELS)
    embedding_model = build_siamese_model(input_shape)
    embedding_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

    save_embeddings_from_signatures()

    app.run(debug=True)
