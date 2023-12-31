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

        n_pixels = np.array(Image.open(self.image_path)).shape[0]**2

        pixel_values = np.array(list(imrgb.getdata()))/255

        return pixel_values, n_pixels

    def fibonacci_sphere(self):

        """
        Evenly distribute pixels of an image along a unit sphere using fibonacci
        method

        Parameters:
        -----------
        sphere: Boolean
        If `True`, show fibonacci sphere

        """
        pixel_values, n_pixels = self.get_pixels()
        n_pixels_width = math.sqrt(n_pixels)

        points = []
        phi = math.pi * (math.sqrt(5) - 1)          # golden angle

        for i in range(n_pixels): 
            y = 1 - (i / float(n_pixels - 1)) * 2   # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)           # radius at y

            theta = phi * i                         # golden angle increment

            x = math.cos(theta) * radius 
            z = math.sin(theta) * radius 

            points.append((x, y, z))

        """
        points_valid = True
        for i in points: 
            norm = math.sqrt(i[0]**2 + i[1]**2 + i[2]**2)
            if norm < 0.99999999999: 
                points_valid = False

        if points_valid == False:
            raise ValueError('Vectors on Bloch Sphere do not have a norm of 1')
        """

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

        print(pixel_points)

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
            print(f"{len(pixel_points)} total pixels")

        return pixel_points

    def image_from_sphere(self, file_name):

        """
        Creates Image from Pixel Positions on Bloch Sphere

        Parameters: 
        -----------
        file_name : String
        name of decrypted file with extension
        
        """
        
        pixel_points = self.align_bloch()
        im = Image.open(self.image_path, 'r')
        im = im.convert("RGB")

        # Ensure `pixel_points` is y-decreasing
        # -------------------------------------
        y_points = np.array([])
        for i in pixel_points:
            y_points = np.append(y_points, [i[0][1]], axis = 0)

        y_check = np.sort(y_points)[::-1]
        if np.array_equal(y_points, y_check) != True:
            raise ValueError('`pixel_points` not y-decreasing')
        # -----------------------------------------------------
        
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
        image_final.save(f"{file_name}")

        return

    def align_bloch(self): 

        """
        Realigns given pixel-rgb list so that it is y-decreasing
        (0, 1, 0) --> (0, -1, 0)
        """
        
        pixel_points = self.fibonacci_sphere()
        pixel_values, n_pixels = self.get_pixels()
        
        # getting just the y-coordinates for each RGB pixel position
        pixel_y_values = np.array([])
        for i in range(len(pixel_points)):
            pixel_y_values = np.append(pixel_y_values, pixel_points[i][0][1])

        # reverse sorting them (y-decreasing)
        ordered_indices = np.flip(np.argsort(pixel_y_values))
        ordered_pixel_points = np.empty(shape = (n_pixels, 2, 3))

        # making new pixel_points matrix sorted y-decreasing
        index = 0 
        for i in ordered_indices: 
            ordered_pixel_points[index] = pixel_points[i]
            index += 1

        return ordered_pixel_points






im = Bloch_Scatter(image_path = "/Users/devaldeliwala/desktop/test.png", 
                   show_sphere = False
)
im.image_from_sphere('output3.png')



