from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from tensorflow import keras
from keras import Layer
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


# class L1Dist(Layer):
#     def __init__(self,**kwargs):
#         super().__init__()
#     def call(self, input_embedding, validation_embedding):
#         return tf.math.abs(input_embedding - validation_embedding)

class CamApp(App):
    def build(self):
        self.img1=Image(size_hint=(1,.8))
        self.button=Button(text="Verify",on_press=self.verify,size_hint=(1,.1))
        self.verification_label=Label(text="Verification Uninitiated", size_hint=(1,.1))
        
        layout=BoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        self.model=tf.keras.models.load_model('siamesemodel5.keras',custom_objects={'L1Dist':L1Dist})

        self.capture=cv2.VideoCapture(0)
        Clock.schedule_interval(self.update,1.0/33.0)
        return layout

    def update(self, *args):
        ret,frame=self.capture.read()
        frame=frame[200:450,200:450,:]
        buf=cv2.flip(frame,0).tostring()
        img_texture=Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        img_texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
        self.img1.texture=img_texture

    def preprocess(self,file_path):
        byte_img=tf.io.read_file(file_path)
        img=tf.io.decode_jpeg(byte_img)
        img=tf.image.resize(img,(105,105))
        img=img/255
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        return img
    
    def verify(self,*args):

        detection_threshold=0.6
        verification_threshold=0.55
        SAVE_PATH=os.path.join('Application_data','Input_image','Input_image.jpg')
        ret,frame=self.capture.read()
        frame=frame[200:450,200:450,:]
        cv2.imwrite(SAVE_PATH,frame)

        results=[]
        for image in os.listdir(os.path.join('Application_data','Verification_images')):
            input_img=self.preprocess(os.path.join('Application_data','Input_image','Input_image.jpg'))
            validation_img=self.preprocess(os.path.join('Application_data','Verification_images',image))
            input_img = tf.expand_dims(input_img, axis=0)  # Shape: (1, 105, 105, 3)
            validation_img = tf.expand_dims(validation_img, axis=0)  # Shape: (1, 105, 105, 3)
            result = self.model.predict([input_img, validation_img],verbose=0)
            results.append(result)
        detection=np.sum(np.array(results)>detection_threshold)
        verification=detection/len(os.listdir(os.path.join('Application_data','Verification_images')))
        verified=verification>verification_threshold

        self.verification_label.text='Verified' if verified==True else 'Unverified'

        Logger.info(results)
        Logger.info(np.sum(np.array(results)>0.05))
        Logger.info(np.sum(np.array(results)>0.1))
        Logger.info(np.sum(np.array(results)>0.2))
        Logger.info(np.sum(np.array(results)>0.4))
        Logger.info(np.sum(np.array(results)>0.6))
        Logger.info(np.sum(np.array(results)>0.8))

        return results, verified

if __name__=='__main__':
    CamApp().run()