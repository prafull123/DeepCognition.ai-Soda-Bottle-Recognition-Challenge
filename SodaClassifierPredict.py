# This is the saved model which I trained using convolution neural networks in the other .py file
# We can now simply load that model

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json,model_from_yaml
import numpy


# load YAML and create model
yaml_file = open('Sodaclassifier.yaml', 'r')
loaded_model_yaml = yaml_file.read()


SodaClassifier = model_from_yaml(loaded_model_yaml)
# load weights into new model
SodaClassifier.load_weights("Sodaclassifier.h5")
print("Loaded model from disk")


# Making new predictions

import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

test_image = image.load_img('Soda_Bottles/single_prediction/Mountain_Dew.jpg', target_size = (64, 64))
#test_image = image.load_img('Soda_Bottles/single_prediction/PepsiDiet.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = SodaClassifier.predict(test_image)
import matplotlib.image as mpimg
import numpy as np
img=mpimg.imread('Soda_Bottles/single_prediction/Mountain_Dew.jpg')
imgplot = plt.imshow(img)

print (result)
print (training_set.class_indices)

yaml_file.close()


