# AI_Programming_with_Python_Nanodegree_Program

Created image classifier to identify dog breeds

An image classification application using a deep learning model, convolutional neural network(CNN). CNNs work particularly well for detecting features in images like colors, textures, and edges; then using these features to identify objects in the images. Have used a CNN that has already learned the features from a giant dataset of 1.2 million images called ImageNet. There are different types of CNNs that have different structures (architectures) that work better or worse depending on your criteria. With this application we used all the three different architectures (AlexNet, VGG, and ResNet) and determined VGG was thr best for this application.

#AlexNet Results:

The CNN architecture used is: alexnet
n_of_images : 40
n_of_dog_images : 30
n_of_non_dog_images : 10
n_correct_dog_breed : 24
n_correct_dog_matches : 30
n_correct_non_dog_matches : 10
n_correct_classified_pet_images : 30
pct_correct_classified_dogs : 100.0
pct_correct_classified_non_dogs : 100.0
pct_correct_classified_dog_breed : 80.0
pct_correct_classified_pet_images : 75.0

** Total Elapsed Runtime: 0:0:3

#VGG Results:
The CNN architecture used is: vgg
n_of_images : 40
n_of_dog_images : 30
n_of_non_dog_images : 10
n_correct_dog_breed : 28
n_correct_dog_matches : 30
n_correct_non_dog_matches : 10
n_correct_classified_pet_images : 36
pct_correct_classified_dogs : 100.0
pct_correct_classified_non_dogs : 100.0
pct_correct_classified_dog_breed : 93.33333333333333
pct_correct_classified_pet_images : 90.0

** Total Elapsed Runtime: 0:0:33

#ResNet Results:

The CNN architecture used is: resnet
n_of_images : 40
n_of_dog_images : 30
n_of_non_dog_images : 10
n_correct_dog_breed : 27
n_correct_dog_matches : 30
n_correct_non_dog_matches : 9
n_correct_classified_pet_images : 33
pct_correct_classified_dogs : 100.0
pct_correct_classified_non_dogs : 90.0
pct_correct_classified_dog_breed : 90.0
pct_correct_classified_pet_images : 82.5

** Total Elapsed Runtime: 0:0:5
