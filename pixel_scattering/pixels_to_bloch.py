from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

class Bloch_Scatter:
    def __init__(self, image_path, show_sphere): 
        
        """
        Parameters: 
        -----------
        image_path   : String
        path to image 

        show_sphere  : Boolean
        if True: show bloch sphere with pixel colors

        """

        self.image_path  = image_path
        self.show_sphere = show_sphere

    def get_pixels(self): 

        """
        Gets RGB information for each pixel from Image

        """

        im = Image.open(self.image_path, 'r')
        imrgb = im.convert("RGB")

        pixel_values = np.array(list(imrgb.getdata()))/255

        return pixel_values

    def fibonacci_sphere(self):

        """
        Evenly distribute pixels of an image along a unit sphere using fibonacci
        method

        Parameters:
        -----------
        sphere: Boolean
        If `True`, show fibonacci sphere

        """

        im = np.array(Image.open(self.image_path))
        
        n_pixels_width = im.shape[0]
        n_pixels = n_pixels_width**2

        points = []
        phi = math.pi * (math.sqrt(5) - 1)          # golden angle

        for i in range(n_pixels): 
            y = 1 - (i / float(n_pixels - 1)) * 2   # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)           # radius at y

            theta = phi * i                         # golden angle increment

            x = math.cos(theta) * radius 
            z = math.sin(theta) * radius 

            points.append((x, y, z))

        pixel_values = self.get_pixels()

        pixel_points = np.array([           # Define 2D Matrix 
            [points[0], pixel_values[0]]    # Initialize Matrix   
        ])

        for i in range(len(points)-1): 
            pixel_points = np.append(pixel_points, 
                                     [[points[i+1], pixel_values[i+1]]], 
                                     axis = 0
            )       
                    # [[point on Fibonacci sphere, pixel RGB value], ...]]
                    # Top-Left pixel at (0, 1, 0)

        if self.show_sphere == True:
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d')

            for i in range(len(points)):
                ax.scatter(pixel_points[i][0][0], 
                    pixel_points[i][0][1],
                    pixel_points[i][0][2],
                    c = pixel_points[i][1].reshape(1, -1),
                    marker = '+'
                )
    
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('$z$')

            plt.show()
            print(len(pixel_points))

        return pixel_points

    def decrypt(self, file_name):

        """
        Creates Image from Pixel Positions on Bloch Sphere

        Parameters: 
        -----------
        file_name : String
        name of decrypted file without extension
        
        """
        
        pixel_points = self.fibonacci_sphere()
        im = Image.open(self.image_path, 'r')
        im = im.convert("RGB")
        
        rgb_list = np.array([pixel_points[0][1]*255])

        for i in range(len(pixel_points)-1):
            rgb_list = np.append(rgb_list, 
                                 [pixel_points[i+1][1]*255], 
                                 axis = 0
                       )

        rgb_list = rgb_list.astype(int)
        rgb_list = tuple(map(tuple, rgb_list))

        image_final = Image.new(im.mode, im.size)
        image_final.putdata(rgb_list)
        image_final.save(f"{file_name}.png")

        return

"""
im = Bloch_Scatter(image_path = "/Users/devaldeliwala/desktop/bracket.png", 
                   show_sphere = True
)
im.decrypt("final2")
"""


