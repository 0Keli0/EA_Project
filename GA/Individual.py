# 1.- numpy matrix representation
import numpy as np
# 2.- random to random values
import random
# 3.- PIL to read, draw and operate with images
from PIL import Image,ImageDraw
#new fitness Delta E
import colour

from IPython.display import Image as dis
import matplotlib.pyplot as plt

import matplotlib as mpl
#mpl.rcParams['figure.figsize'] = (15,10) #resize the image

#load the photo

class Individual:
    """
    Individual defined by: 
        -lenght    -> the lenght of the image target
        -width     -> the width of the image target
        -fitness   -> the error between the image of the individual and the image target
        -polygon_n -> the maximun number of polygons that is going to be used in case where we create a random image from this individual
        -image     -> the image that define the individual as an Image.Image from Pillow
        -array     -> the image that define the individual as an np.ndarray from NumPy

    """
    def __init__(self, length, width, polygon=6, img=None):  
        self.length = length
        self.width = width
        self.fitness = float('inf')
        self.polygon_n = polygon
        if(isinstance(img,Image.Image)): #In cases where it receive an image as an obj Image.Image (Pillow)
            self.image = img
            self.array = np.array(img,dtype=np.uint8)    
            
        elif(isinstance(img,np.ndarray)): #In case where it receive an array as image (Numpy)
            self.image = Image.fromarray(img.astype(np.uint8))
            self.array = img.astype(np.uint8)
        else:   #in cases where there is no img gave
            self.image = None
            self.array = None
            self.create_random_image()

    #Method that create set the properties "image" and "array" generating a random image with a random number of polygons between [3,polygon_n]
    def create_random_image(self):
        polygon_num = random.randint(3,self.polygon_n) #Generate from 3 to polygon_n polygons
        region = (self.length+self.width)//8 #With this we will give the chance to create points out of the margin but not so far away
        #this will allow to create more different geometric structures that could be good for the design
        
        #Generate a new object Image with the size of the problem and
        #  with a random color for the background gived as a String code
        img = Image.new(mode="RGBA", size=(self.length,self.width), color=self.random_string_color()) 
        for i in range(polygon_num):
            num_points = random.randint(3,6) #from polygon of 3 to 6 points
            x = random.randint(0,self.length)
            y = random.randint(0,self.width)
            #Create a list of tuple_points considering the number of points that conform the polygon
            #The values of the x/y dimension are selected randomly with the margin bounds, and we also 
            # let the points be created outside the margin within a given "region" which makes them not stray too far away.
            xy = [(random.randint(x-region,x+region),random.randint(y-region,y+region)) 
                   for _ in range(num_points)]
            #After create the points, giving to the polygon method the list of dots as parameters and the random color 
            # that the PIL has to create random colors
            img1 = ImageDraw.Draw(img)
            img1.polygon( xy,fill=self.random_string_color())

        self.image = img
        self.array = np.array(img)

    #Method fitness which provide the fitness that the Individual has as an error 
    # between the Individual image and the image target
    def get_fitness(self,target):
            self.fitness=np.mean( colour.difference.delta_e.delta_E_CIE1976(target,self.array ) ) # type: ignore
    
    #Method that return a list of Individuals which are the divisions of this Individual. The dimension of the division is defined by the
    # parameter next_block_size which define the (width,lenght) proportion that is going to be took from the photo. At the end,
    # multiplying this 2 portions given we can know in how many pieces we are diving this Individual. The default setting is 
    # in 9 pieces.
    def div_in_blocks(self,next_block_size = (3,3)):
        w_size, l_size= next_block_size
        img = self.__copy__().array
        blocks = []
        b_w_size, b_l_size = self.width // w_size , self.length // l_size
        for iw in range(w_size):
            for il in range(l_size):
                x = b_w_size * iw
                y = b_l_size * il
                if(isinstance(img,np.ndarray)):
                    block = img[int(x):int(x+b_w_size), int(y):int(y+b_l_size)]
                    new_Ind = Individual(b_l_size,b_w_size,img=block)
                    blocks.append(new_Ind)
        return blocks

    #Method that allows us make a copy of an Individual object
    def __copy__(self):
        return Individual(self.length,self.width,img=self.image)
    
    @staticmethod
    #Static Funtion that give as a random string that suite a color codification
    def random_string_color(): 
        return "#"+''.join([random.choice('0123456789ABCDEF') 
                            for j in range(6)])
    

'''
original_image = 'gioconda.jpg'
img = Image.open(original_image).convert('RGBA')
print(isinstance(img,Image.Image))
target = np.array(img)
print(type(target))

print(isinstance(target,np.ndarray))
target_w,target_l,target_d = target.shape'''
#print(target.shape[:2])    

#ind = Individual(target_l,target_w)
#dis(ind.image)
#plt.imshow(ind.image)
#plt.show()
        



