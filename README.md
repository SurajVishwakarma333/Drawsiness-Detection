# Drawsiness 🥱-Detection⚠️
Drowsiness detection is a safety technology that can prevent accidents that are caused by drivers who fell asleep while driving. The objective of this intermediate Python project is to build a drowsiness detection system that will detect that a person's eyes are closed for a few seconds.



![img](https://user-images.githubusercontent.com/101108540/171376466-235c00aa-d25e-49bb-8562-863b780fbc4c.jpeg)


# HOW TO CREATE A PROJECT ? LETS START..


• Install this all 8 library in anaconda promt before starting the project. 


     1. pip install tensorflow
     2. pip install tensorflow-gpu
     3. pip install opencv-python
     4. pip install opencv-contrib-python
     5. pip install matplotlib
     6. pip install numpy
     7. pip install pygame
     8. pip install -U pygame
                    
                    

• Step 1 : Create new ipynb(Interactive PYthon NoteBook)which means Jupyter notebook File and rename as Data Preparation.



     import os
     import shutil
     import glob
     from tqdm import tqdm
            
     //dividing the photos in open eyes and close eyes two different folder.
            
     Raw_DIR=r'E:\SurajPython\eye\mrlEyes_2018_01'
     for dirpath, dirname, filenames in os.walk(Raw_DIR):
       for i in tqdm([f for f in filenames if f.endswith('.png')]):
         if i.split('_')[4]=='0':
           shutil.copy(src=dirpath+'/'+i,dst=r'E:\SurajPython\eye\Prepared_Data\Close Eyes')    
         elif i.split('_')[4]=='1':
           shutil.copy(src=dirpath+'/'+i,dst=r'E:\SurajPython\eye\Prepared_Data\Open Eyes')
                  
      //we have done and devided the photos in open eyes and close eyes two different folder.
      
      
      
      
• Step 2 : Create new ipynb(Interactive PYthon NoteBook)which means Jupyter notebook File and rename as Model Training.
      
      
      
      
     import tensorflow as tf
     from tensorflow.keras.applications import InceptionV3
     from tensorflow.keras.models import Model
     from tensorflow.keras.layers import Dropout, Input, Flatten, Dense, MaxPooling2D
     from tensorflow.keras.optimizers import Adam
     from tensorflow.keras.preprocessing.image import ImageDataGenerator

     tf.test.is_gpu_available()    //checking for GPU is available or not in system

     batchsize=8

     train_datagen=ImageDataGenerator(rescale=1./225,rotation_range=0.2,shear_range=0.2,zoom_range=0.2, width_shift_range=0.2,height_shift_range=0.2,                          validation_split=0.2)
     train_data= train_datagen.flow_from_directory(r'E:\SurajPython\eye\Prepared_Data\train',target_size=                                                              (80,80),batch_size=batchsize,class_mode='categorical',subset='training')
     validation_data= train_datagen.flow_from_directory(r'E:\SurajPython\eye\Prepared_Data\train',target_size=                                                                (80,80),batch_size=batchsize,class_mode='categorical',subset='validation')

     test_datagen = ImageDataGenerator(rescale=1./255)
     test_data = test_datagen.flow_from_directory(r'E:\SurajPython\eye\Prepared_Data\test',target_size=(80,80),batch_size=batchsize,class_mode='categorical')


     //this is deep learning
     bmodel = InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(80,80,3)))
     hmodel = bmodel.output
     hmodel = Flatten()(hmodel)
     hmodel = Dense(64, activation='relu')(hmodel)
     hmodel = Dropout(0.5)(hmodel)
     hmodel = Dense(2,activation= 'softmax')(hmodel)
     //space here
     model = Model(inputs=bmodel.input, outputs= hmodel)
     for layer in bmodel.layers:
          layer.trainable = False

     //To see the summary of model architecture  where we use InceptionV2 model as a architecture(optional)
     model.summary()

     from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau

     checkpoint = ModelCheckpoint(r'E:\SurajPython\eye\models',
                            monitor='val_loss',save_best_only=True,verbose=3)
     earlystop = EarlyStopping(monitor = 'val_loss', patience=7, verbose= 3, restore_best_weights=True)
     learning_rate = ReduceLROnPlateau(monitor= 'val_loss', patience=3, verbose= 3, )
     callbacks=[checkpoint,earlystop,learning_rate]




     model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])
     model.fit_generator(train_data,steps_per_epoch=train_data.samples//batchsize,
                   validation_data=validation_data,
                   validation_steps=validation_data.samples//batchsize,
                   callbacks=callbacks,
                    epochs=5)            //here we can set echo number as per our requirement


     ##From here we are going for Model Evaluation

     //optional to run and can stop it
     acc_tr, loss_tr = model.evaluate_generator(train_data)
     print(acc_tr)
     print(loss_tr)

     //optional to run and can stop it
     acc_vr, loss_vr = model.evaluate_generator(validation_data)
     print(acc_vr)
     print(loss_vr)

     //optional to run and can stop it
     acc_test, loss_test = model.evaluate_generator(test_data)
     print(acc_tr)
     print(loss_tr)
 
 
 
• Step 3 : Create new ipynb(Interactive PYthon NoteBook)which means Jupyter notebook File and rename as main.


    import cv2
    import pygame
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import numpy as np
    from pygame import mixer

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    model = load_model(r'E:\SurajPython\eye\models\model.h5')     //here copy the path of models file and write after that \model.h5

    mixer.init()
    sound= mixer.Sound(r'E:\SurajPython\alarm.wav')
    cap = cv2.VideoCapture(0)
    Score = 0
    while True:
      ret, frame = cap.read()
      height,width = frame.shape[0:2]
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces= face_cascade.detectMultiScale(gray, scaleFactor= 1.2, minNeighbors=3)
      eyes= eye_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=1)
    
     cv2.rectangle(frame, (0,height-50),(200,height),(0,0,0),thickness=cv2.FILLED)
    
     for (x,y,w,h) in faces:
        cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h), color= (255,0,0), thickness=3 )
         
     for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color= (255,0,0), thickness=3 )
        
        # preprocessing steps
        eye= frame[ey:ey+eh,ex:ex+w]
        eye= cv2.resize(eye,(80,80))
        eye= eye/255
        eye= eye.reshape(80,80,3)
        eye= np.expand_dims(eye,axis=0)
        # preprocessing is done now model prediction
        prediction = model.predict(eye)
        
        # if eyes are closed
        if prediction[0][0]>0.30:
            cv2.putText(frame,'closed',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            Score=Score+1
            if(Score>15):
                try:
                    sound.play()
                except:
                    pass
            
        # if eyes are open
        elif prediction[0][1]>0.90:
            cv2.putText(frame,'open',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)      
            cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            Score = Score-1
            if (Score<0):
                Score=0
            
        
     cv2.imshow('frame',frame)
     if cv2.waitKey(33) & 0xFF==ord('q'):
        break
        
     cap.release()
     cv2.destroyAllWindows()
     
     
     
     
# That's it! 😊    
