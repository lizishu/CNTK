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
vgg_test4.py: adjusted vgg19, error = 8.48%  
              
