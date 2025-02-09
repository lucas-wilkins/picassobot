import cv2
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

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

    def __init__(self,
                 x0: float = 210.0/2, y0: float = 297.0/2,
                 x_size: float = 180.0, y_size: float = 250.0,
                 z_off: float = 10.0, z_on: float = 0.0,
                 speed: float = 100.0, pen_lift_speed: float = 100.0,
                 resolution=50_000, detail=40, brightness_scale=255):

        self.resolution = resolution
        self.detail = detail
        self.brightness_scale = brightness_scale

        self.x0 = x0
        self.y0 = y0
        self.x_size = x_size
        self.y_size = y_size

        self.z_off = z_off
        self.z_on = z_on

        self.speed = speed
        self.pen_lift_speed = pen_lift_speed

    @abstractmethod
    def base_path(self):
        pass

    def path(self):
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

    @abstractmethod
    def elaboration(self, position, value):
        pass

    def vectorise(self, im, show: Optional[str]="Vectorising..."):

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

        im = adjust_contrast_brightness(im, contrast=2)
        if show:
            cv2.imshow(show, im)
            cv2.waitKey(0)

        im = self.brightness_scale - im

        s, x, y, nx, ny = self.path()

        # scale is one more than the maximum index, some bodges around that
        x_px = np.array((x+0.5)*scale*0.99999, dtype=int)
        y_px = np.array((y+0.5)*scale*0.99999, dtype=int)
        values = im[scale-1-y_px, x_px]

        offsets = self.elaboration(s, values)

        x_adjusted = x + offsets*nx
        y_adjusted = y + offsets*ny

        if show:
            plt.plot(x_adjusted, y_adjusted)
            plt.show()

        return x_adjusted, y_adjusted

    def gcode(self, im, show="Vectorising..."):
        x_raw, y_raw = self.vectorise(im, show)

        output = []

        x_home = self.x0
        y_home = self.y0

        x = x_raw * self.x_size + self.x0
        y = y_raw * self.y_size + self.y0

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

        # Home
        output.append("")
        output.append("; Home")
        output.append(f"g0 x{x_home} y{y_home} f{self.speed}")

        return "\n".join(output)

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
        angles = positions * (2*self.detail*np.pi)
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



# List of different drawing methods, triangle is shit

methods = [SpiralSine,
           # SpiralTriangle,
           SnekSine,
           # SnekTriangle
]

if __name__ == "__main__":
    from load_test_image import load_image

    im = load_image()

    for cls in methods:

        vectoriser = cls()

        # vectoriser.show_elaboration()

        x, y = vectoriser.vectorise(im, show=None)
        plt.plot(x, y)
        plt.show()

        with open(f"test_{cls.__name__}.gcode", 'w') as file:
            file.write(vectoriser.gcode(im, show=None))
            file.write("\n")