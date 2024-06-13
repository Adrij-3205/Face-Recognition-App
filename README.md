# Face-Recognition-App
An app which will show if an user is verified or not based on real time webcam image of that person.
Concepts used-
1. Used siamese neural network for making predictions.
2. There are three sets of images: anchor images, positive images, negative images.
3. For training, the anchor images as well as the positive images are of the person for whom the app should display 'Verified' and the negative images are from the labelled faces in the wild dataset.
4. The anchor images are compared with positive images and the label is assigned as 1 and for negative images it is assigned as 0.
5. The app is made using kivy, which allows a customised simple design.
