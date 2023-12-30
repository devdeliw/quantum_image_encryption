from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

class Bloch_Scatter:

    def __init__(self, image_path): 
        
        """
        Parameters: 
        -----------

        image_path   : String
        path to image 

        """

        self.image_path = image_path

    def get_pixels(self): 

        im = Image.open(self.image_path, 'r')
        imrgb = im.convert("RGB")

        pixel_values = list(imrgb.getdata())

    def fibonacci_sphere(self, sphere):

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


        print(f"This image is {n_pixels_width} x {n_pixels_width} pixels ")

        points = []
        phi = math.pi * (math.sqrt(5) - 1)          # golden angle

        for i in range(n_pixels): 
            y = 1 - (i / float(n_pixels - 1)) * 2   # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)           # radius at y

            theta = phi * i                         # golden angle increment

            x = math.cos(theta) * radius 
            z = math.sin(theta) * radius 

            points.append((x, y, z))
            print(points)

        pixel_points = np.array([           # Define 2D Matrix 
            [points[0], pixel_values[0]]    # Initialize Matrix   
        ])

        for i in range(len(points)-1): 
            pixel_points = np.append(pixel_points, 
                                     [[points[i+1], pixel_values[i+1]]], 
                                     axis = 0
            )       # Matrix has [point on fibonacci sphere, pixel RGB
                    # for every pixel on the sphere
                    # Top-Left pixel at (0, 1, 0)

        if sphere == True:

            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d')

            for i in range(len(points)):

                ax.scatter(points[i][0],
                           points[i][1],
                           points[i][2],
                           c = 'black', marker = '.'
                )

            plt.show()


        return pixel_points

im = Bloch_Scatter("/Users/devaldeliwala/desktop/bracket.png")
im.fibonacci_sphere()
