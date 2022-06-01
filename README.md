# Drawsiness 🥱-Detection ⚠️
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
      
