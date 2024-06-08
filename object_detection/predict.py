from typing import Any

import json
import numpy as np
from ultralytics import YOLO
from cog import BasePredictor, Input, Path
# from tensorflow.keras.applications.resnet50 import (
#     ResNet50,
#     decode_predictions,
#     preprocess_input,
# )
# from tensorflow.keras.preprocessing import image as keras_image


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = YOLO("best.pt")
        #ResNet50(weights="resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    # Define the arguments and types the model takes as input
    def predict(self, image: Path = Input(description="Image to detect")) -> Any:
        """Run a single prediction on the model"""
        # Preprocess the image
        #img = keras_image.load_img(image, target_size=(224, 224))
        #x = keras_image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        # Run the prediction
        preds = self.model.predict(image)
        classes = json.dumps(preds[0].names)
        data = preds[0].boxes.data.numpy()
        # Return the top 3 predictions
        print(classes, data)
        return classes, data # decode_predictions(preds, top=3)[0]
