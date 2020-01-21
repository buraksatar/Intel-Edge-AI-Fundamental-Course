import cv2
import numpy as np

def pose_estimation(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)
    # TODO: Preprocess the image for the pose estimation model
    new_size = (456,256)
    # since resize takes width first we need to exchange the
    # normal heightxwidth convention
    resized_image = cv2.resize(preprocessed_image, new_size)
    
    transposed_image = resized_image.transpose((2,0,1))
    
    reshaped_image = transposed_image.reshape(1,3,256,456)
    #print(reshaped_image.shape)
    
    return reshaped_image


def text_detection(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related text detection model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)
    # TODO: Preprocess the image for the pose estimation model
    new_size = (1280,768)
    # since resize takes width first we need to exchange the
    # normal heightxwidth convention
    resized_image = cv2.resize(preprocessed_image, new_size)
    
    transposed_image = resized_image.transpose((2,0,1))
    
    reshaped_image = transposed_image.reshape(1,3,768,1280)
    #print(reshaped_image.shape)
    return reshaped_image


def car_meta(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related car metadata model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    
    preprocessed_image = np.copy(input_image)
    # TODO: Preprocess the image for the pose estimation model
    new_size = (72,72)
    resized_image = cv2.resize(preprocessed_image, new_size)
    
    transposed_image = resized_image.transpose(2,0,1)
    
    reshaped_image = transposed_image.reshape(1,3,72,72)
    #print(reshaped_image.shape)
    return reshaped_image
