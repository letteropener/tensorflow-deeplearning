import matplotlib.pyplot as plt
plt.figure(1)
plt.plot([2,4,8,16,32,64,128,256,512,1024,2048],[0.009132, 0.008610, 0.025843, 0.080485, 0.674794, 4.135585,
          30.981771,195.956573,453.597015,803.031799,3735.943359], 'r--', label='Square Matrix(size= power of 2)')
plt.plot([2,10,20,50,100,200,500,1000,2000],[0.008242, 0.037389, 0.178067, 2.020147, 14.415960, 81.643959,
          308.601868,648.789185,1722.324463], 'bo-', label='Square Matrix(size= Multiple of 10)')
plt.plot([2,4,8,16,32,64,128,256,512,1024,2048],[0.007870, 0.009060, 0.015865, 0.074616, 0.534399, 6.033558,
                                                 25.053070,98.223709,175.865173,271.456482,820.252136], 'g^-', label='Not Square Matrix(ex:1024X1024X1023)')
plt.ylabel('Speed up (Times X)')
plt.xlabel('# of threads per block')
plt.legend()
plt.title('Performance speed up GPU/CPU')

plt.figure(2)
plt.plot([64,256,400,625,1024],[432.175140,453.597015,431.289551,398.528778,449.086975], 'r--', label='Avg speed up (Matrix size = 512 * 512 * 512)')
plt.plot([64,256,400,625,1024],[653.1095101,803.031799,728.4826635,664.6416,768.0069], 'bo-', label='Avg speed up (Matrix size = 1024 * 1024 * 1024)')
plt.plot([64,256,400,625,1024],[2417.259766,3735.943359,3468.828857,2966.204102,3487.605957], 'g^-', label='Avg speed up (Matrix size = 2048 * 2048 * 2048)')
plt.ylabel('Speed up (Times X)')
plt.xlabel('# of threads per block')
plt.legend()
plt.title('Performance speed up GPU/CPU with BLOCK_SIZE')
plt.show()
