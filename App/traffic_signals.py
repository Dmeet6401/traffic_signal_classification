import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import tensorflow as tf


st.header("Intel Image Classifier")
def main():
    file_uploaded = st.file_uploader("Choose the file" ,type = ['Jpg', "png","jpeg"])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        # image = cv2.resize(image,(150,150))
        figure = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)
def predict_class(image):
    model = tf.keras.models.load_model(r"C:\Users\admin\Desktop\DX codes\computer vision\Task 3\models\traffic_vgg16_120.h5")
    # shape = ((299,299,3))
    # model = tf.keras.Sequential(hub[hub.KerasLayer(model,input_shape = shape)])
    # test_image=preprocessing.image.img_to_array(image)
    test_image = np.array(image)
    test_image = cv2.resize(test_image,(150,150))
    test_image=test_image/255.0
    test_image = np.expand_dims(test_image,axis = 0)
    class_names = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '6', '7', '8', '9']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    # scores = max(scores.numpy()) to get maximum accuracy only
    image_class = class_names[np.argmax(scores)]
    # predicted = class_names[np.argmax(model.predict(test_image)[0])]
    result = st.write("The image uploaded is: {}".format(image_class))
    j = 0
    for i in scores:
        st.write("probability of  {} is {:.2f} ".format(class_names[j], (i) * 100 ))
        j+=1
    
    # prob =st.write("probability of image is: {}".format((scores) * 100 )) 
    return result
    
if __name__ == "__main__":
    main()