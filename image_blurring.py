import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image 
import time
from numba import njit

@njit
def blurfilter(in_img, out_img, kernel, kernel_size): 
    pxc = 0
    for c in range(3): 
        for  x in range(in_img.shape[1]):
           pxc += 1
           for y in range(in_img.shape[0]): 
              val = 0 
              for i in range(-(kernel_size//2),(kernel_size//2)+1): 
                 for j in range(-(kernel_size//2),(kernel_size//2)+1): 
                   if (x+i < in_img.shape[1]) and (x+i >= 0) and \
                      (y+j < in_img.shape[0] ) and (y+j >=0 ): 
                        val += (int(in_img[y+j,x+i,c] )) * kernel[i, j]
              out_img[y,x,c] = val

@njit 
def get_kernel(length, sigma):
    ax = np.linspace(-(length-1)/2.0, (length-1)/2.0, length)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    normalize_kernel = kernel / np.sum(kernel)
    print(normalize_kernel)
    return normalize_kernel


def main():                  
    img = np.array(Image.open('noisy1.jpg')) 
    print(img.shape) 
    imgblur= img.copy()
    #start timing
    start_time = time.time()
    kernel_size = 15
    kernel = get_kernel(kernel_size,1)
    blurfilter(img, imgblur, kernel, kernel_size)
    # end timing
    stop_time = time.time()
    print("%s seconds" % (stop_time - start_time))
    # Display and save blurred image 
    fig = plt.figure() 
    ax = fig.add_subplot(1, 2, 1) 
    imgplot = plt.imshow(img) 
    ax.set_title('Before') 
    ax = fig.add_subplot(1, 2, 2) 
    imgplot = plt.imshow(imgblur) 
    ax.set_title('After') 
    img2= Image.fromarray(imgblur) 
    img2.save('blurred.jpg')
    plt.show()

if __name__ == "__main__":
    main()
