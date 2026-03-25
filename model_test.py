from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model("best_model.keras")

# image you want to load 
img = image.load_img("/home/tabish/unseen data/test10.jpg",target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array) #type: ignore

if pred[0][0] > 0.5:
    label="Dog"
    
else:
    label="Cat"
    
print(f"prediction:{pred[0][0]:.4f},label:{label}")    
    

