#------------  Use the .png of the folder to create the .gif
import os
import imageio

dirName1 = './Data gif/'
dirName2 = './Gaussian gif/'

dirs = os.listdir(dirName2) #--- change dirName 1 ,2
dirs.sort(key=lambda x:int(x[:-4]))
frames = []

gif_name1 = 'Data_gif.gif'
gif_name2 = 'Gaussian_gif.gif'
for file in dirs:    
    frames.append(imageio.imread(dirName2 + file)) #-- change dirName 1,2
    imageio.mimsave(gif_name2, frames, 'GIF', duration = 0.1) #-- change gif_name 1 ,2
    print (file) 
    

    











