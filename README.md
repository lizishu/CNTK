
# CNTK
learning CNTK  

# AlexNet_demo  
AlexNet_demo.py  : error = 10.88%  
Alex_jitter.py  : input add jitter error = 11.48%  

# NIN_demo  
NIN_test.py: error = 36.48%  
NIN_test2.py: change some parameters, error = 28.93%  
NIN_test3.py: change some parameters, error = 27.98%  
NIN_test4.py: Using AveragePooling without dense layer , error = 25.43%  
NIN_test5.py: Using AveragePooling with dense layer, error = 21.20%
NIN_test6.py: remove one 1 * 1 layer, error = 23.07%  

# VGG_demo  
vgg_test1.py: adjusted vgg13, failed to converge  
vgg_test2.py: fixed the bug, reduce learning rate, error = 13.63%  
vgg_test3.py: adjusted vgg16  
*   max_epoch = 40, with bathnormalization, error = 12.87% 
*   max_epoch = 40, without batchnormalization, error = 27.93% 
*   max_epoch = 80, with batchnormalization, error = 8.83%

vgg_test4.py: adjusted vgg19, with batchnormalization, error = 8.48%  

# GoogleNet_demo
## InceptionV1
Inceptionv1_test1.py  
*   max_epoch = 80, lr = AlexNet , error = 32.28%
*   max_epoch = 160, lr = 0.01, every two epoch , lr = lr * 0.94, error =  18.97%
              
## BN_Inception
BN_Inception_test1.py ：add an inception block after 4b , error = 5.86%  
BN_Inception_test2.py : change some parameters, error = 5.58%
