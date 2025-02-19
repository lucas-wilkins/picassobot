from enum import Enum
from typing import Optional
from abc import ABC, abstractmethod

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_contrast_brightness(img, contrast:float=1.0, brightness:int=0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """

    # From a Stack Overflow post

    brightness += int(round(255*(1-contrast)/2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)

class AbstractPathDrawer:
    """ Base class for things that draw paths"""

    def __init__(self,
                 x0: float = 210.0/2, y0: float = 297.0/2,
                 x_size: float = 200.0, y_size: float = 200.0,
                 logo_size: float=40, logo_x: float=210-30, logo_y: float = 30,
                 title_size: float = 100, title_x = 210/2, title_y: float = 297 - 30,
                 z_off: float = 10.0, z_on: float = 0.0,
                 speed: float = 3000.0, pen_lift_speed: float = 1000.0,
                 resolution=50_000, detail=40, brightness_scale=255):

        self.resolution = resolution
        self.detail = detail
        self.brightness_scale = brightness_scale

        self.x0 = x0
        self.y0 = y0
        self.x_size = x_size
        self.y_size = y_size

        self.logo_size = logo_size
        self.logo_x = logo_x
        self.logo_y = logo_y

        self.title_size = title_size
        self.title_x = title_x
        self.title_y = title_y

        self.z_off = z_off
        self.z_on = z_on

        self.speed = speed
        self.pen_lift_speed = pen_lift_speed

    @abstractmethod
    def base_path(self):
        """ Generate the base path to work from"""

    def path(self):
        """ Get extra information about the path"""
        x, y = self.base_path()

        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]

        ds = np.sqrt(dx ** 2 + dy ** 2)

        # Normals - note, scaled to match gap between spirals
        nx = np.concatenate(([0], dy / (ds * self.detail)))
        ny = np.concatenate(([0], -dx / (ds * self.detail)))

        # Whole curve goes between 0 and 1
        s = np.cumsum(ds)
        s /= s[-1]
        s = np.concatenate(([0], s))

        return s, x, y, nx, ny

    def show_base_curve(self):
        x, y = self.base_path()
        print(f"x in [{min(x)}, {max(x)}]")
        print(f"y in [{min(y)}, {max(y)}]")
        plt.plot(x, y)
        plt.show()

    @abstractmethod
    def elaboration(self, position, value):
        """ How to change the path, displacement normal to curve"""

    def vectorise(self, im, show: Optional[str]="Vectorising..."):
        """ Create the path needed to draw an image, scaled to (-0.5,0.5) on each axis"""


        if show:
            cv2.imshow(show, im)
            cv2.waitKey(0)

        # Convert to greyscale
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        if show:
            cv2.imshow(show, im)
            cv2.waitKey(0)

        # Get the square in the centre of the image
        if im.shape[0] > im.shape[1]:
            x_offset = (im.shape[0] - im.shape[1]) // 2
            y_offset = 0
            scale = im.shape[1]
        else:
            x_offset = 0
            y_offset = (im.shape[1] - im.shape[0]) // 2
            scale = im.shape[0]


        im = im.reshape(im.shape[0], im.shape[1], -1)[x_offset:x_offset+scale, y_offset:y_offset+scale, 0]

        if show:
            cv2.imshow(show, im)
            cv2.waitKey(0)

        im = adjust_contrast_brightness(im, contrast=1.2)
        if show:
            cv2.imshow(show, im)
            cv2.waitKey(0)

        im = self.brightness_scale - im

        s, x, y, nx, ny = self.path()

        # scale is one more than the maximum index, some bodges around that
        x_px = np.array((x+0.5)*scale*0.99999, dtype=int)
        y_px = np.array((y+0.5)*scale*0.99999, dtype=int)
        # values = im[y_px, x_px]
        values = im[scale-1-y_px, x_px]

        offsets = self.elaboration(s, values)

        x_adjusted = x + offsets*nx
        y_adjusted = y + offsets*ny

        if show:
            plt.plot(x_adjusted, y_adjusted)
            plt.show()

        return x_adjusted, y_adjusted

    def gcode(self, im, show="Vectorising..."):
        """
        Generate gcode for an image
        """
        x_raw, y_raw = self.vectorise(im, show)

        output = []

        x_home = 0
        y_home = 0

        x = x_raw * self.x_size + self.x0
        y = -y_raw * self.y_size + self.y0

        output.append("")
        output.append("; Main image")

        # Lift pen
        output.append("")
        output.append("; Lift pen")
        output.append(f"g0 z{self.z_off} f{self.pen_lift_speed}")

        # go to first position
        output.append("")
        output.append("; Start position")
        output.append(f"g0 x{x[0]} y{y[0]} f{self.speed}")

        # drop pen
        output.append("")
        output.append("; Drop pen")
        output.append(f"g0 z{self.z_on} f{self.pen_lift_speed}")

        # Main image
        output.append("")
        output.append("; Draw image")
        for xi, yi in zip(x, y):
            output.append(f"g0 x{xi} y{yi} f{self.speed}")

        # Lift
        output.append("")
        output.append("; Lift pen")
        output.append(f"g0 z{self.z_off} f{self.pen_lift_speed}")

        #
        # # Title
        # output.append("")
        # output.append("; Title")
        #
        # output += self.data_curves("title_curve", self.title_size, self.title_x, self.title_y)
        #
        # Logo
        output.append("")
        output.append("; Logo")

        output += self.data_curves("logo_curve", self.logo_size, self.logo_x, self.logo_y)


        # Home
        output.append("")
        output.append("; Home")
        output.append(f"g0 x{x_home} y{y_home} f{self.speed}")

        return "\n".join(output)

    def data_curves(self, file_prefix, size, x0, y0):

        output = []

        for filename in os.listdir("data"):
            if filename.startswith(file_prefix):
                path = np.load(os.path.join("data", filename))

                x = size * path[:, 0] + x0
                y = -size * path[:, 1] + y0

                # go to first position
                output.append("")
                output.append(f"; Start position - {filename}")
                output.append(f"g0 x{x[0]} y{y[0]} f{self.speed}")

                # drop pen
                output.append("")
                output.append("; Drop pen")
                output.append(f"g0 z{self.z_on} f{self.pen_lift_speed}")

                # draw curve
                output.append("")
                output.append("; Curve")
                for xi, yi in zip(x[1:], y[1:]):
                    output.append(f"g0 x{xi} y{yi} f{self.speed}")

                # lift pen
                output.append("")
                output.append("; Lift pen")
                output.append(f"g0 z{self.z_off} f{self.pen_lift_speed}")

        return output

    def show_elaboration(self, n=1000):
        """ Show a plot of the elaboration curve"""

        position = np.linspace(0, 1, self.resolution)[:n]
        values = np.ones(n)

        curve = self.elaboration(position, values)
        plt.plot(position, curve, 'x-')

        plt.show()

class Spiral(AbstractPathDrawer):
    def base_path(self):
        positions = np.sqrt(np.linspace(0.0, 1.0, self.resolution))
        angles = positions * (2*self.detail*np.pi) + np.pi/4
        x = 0.5 * np.sin(angles) * positions
        y = 0.5 * np.cos(angles) * positions

        return x, y

class Snek(AbstractPathDrawer):
    cap_frac = 0.0

    def base_path(self):
        positions = np.linspace(0.0, 1.0, self.resolution) * self.detail
        line_pair_s = positions % 1.0 # raw_x goes between 0 and 1 for each line pair

        line_s = 2*line_pair_s % 1 # Double up the lines
        y_unscaled = 2*positions - line_s # Removes fractional part

        # Reverse every other line
        x = line_s.copy()
        odd = line_pair_s > 0.5
        x[odd] = 1-x[odd]

        # rescale before returning
        y = y_unscaled / (2*self.detail)

        return x-0.5, y-0.5

class Pentagon(AbstractPathDrawer):
    def base_path(self):
        positions = np.sqrt(np.linspace(0.0, 1.0, self.resolution))
        angles = positions * (2*(self.detail-1)*np.pi) + 2*np.pi
        r = (angles + (1 + (4/self.detail)*angles) * np.sin(5*angles)) / (self.detail*np.pi)

        x = np.cos(angles) * r
        y = np.sin(angles) * r

        # times 2 because we want to map to [-0.5, 0.5]
        scale = 2*max([np.max(x), np.max(-x), np.max(y), np.max(-y)])

        x /= scale
        y /= scale

        return x, y


class Square(AbstractPathDrawer):
    def base_path(self):
        positions = np.sqrt(np.linspace(0.0, 1.0, self.resolution))
        angles = positions * (2*(self.detail-1)*np.pi) + 2*np.pi
        r = (angles + (1 + (4/self.detail)*angles) * np.sin(4*angles - np.pi/2)) / (self.detail*np.pi)

        x = np.cos(angles) * r
        y = np.sin(angles) * r

        # times 2 because we want to map to [-0.5, 0.5]
        scale = 2*max([np.max(x), np.max(-x), np.max(y), np.max(-y)])

        x /= scale
        y /= scale

        return x, y

class Sine(AbstractPathDrawer):
    def elaboration(self, position, value):
        return (0.2/self.brightness_scale)*np.sin(0.75*self.resolution*position)*value


class Triangle(AbstractPathDrawer):
    def elaboration(self, position, value):
        # Make a triangle

        output = (0.4*self.resolution) * position

        # Turn into triangle wave
        output %= 2
        over_one = output > 1
        output[over_one] = 2 - output[over_one]

        output *= (1+np.array(value, dtype=float)) / (1+self.brightness_scale)

        # Return scaled and offset
        return (output - 0.5) * 0.3


class SpiralSine(Spiral, Sine):
    pass

class SnekSine(Snek, Sine):
    pass

class SpiralTriangle(Spiral, Triangle):
    pass

class SnekTriangle(Snek, Triangle):
    pass

class PentagonSine(Pentagon, Sine):
    pass

class SquareSine(Square, Sine):
    pass

# # List of different drawing methods, triangle is shit
# methods = [
#             # SpiralTriangle,
#             # SnekTriangle
#             SpiralSine,
#             SnekSine,
#             SquareSine,
#             PentagonSine,
# ]


class Shape(Enum):
    SQUARE = "square"
    CIRCLE = "circle"
    SNEK = "snek"
    STAR = "star"

methods = {
    Shape.SQUARE: SquareSine,
    Shape.CIRCLE: SpiralSine,
    Shape.SNEK: SnekSine,
    Shape.STAR: PentagonSine
}

if __name__ == "__main__":
    from load_test_image import load_image
    from gcode_viewer import view_gcode

    im = load_image()

    for cls in methods:

        vectoriser = cls()

        # vectoriser.show_base_curve()
        # vectoriser.show_elaboration()

        # x, y = vectoriser.vectorise(im, show=None)
        # plt.plot(x, y)
        # plt.show()

        filename = f"test_{cls.__name__}.gcode"

        with open(filename, 'w') as file:
            file.write(vectoriser.gcode(im, show=None))
            file.write("\n")

        view_gcode(filename)