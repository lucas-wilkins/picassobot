import numpy as np
import matplotlib.pyplot as plt

def view_gcode(filename: str, pen_touch_height=5):
    """ View plotter gcode"""

    x = 0
    y = 0
    z = 0

    x_positions = [x]
    y_positions = [y]
    z_positions = [z]

    # Build a list of moves

    with open(filename, 'r') as file:
        for line in file:
            parts = line.split(";") # Remove comments
            code = parts[0].strip()


            if code != "":
                parts = [part.lower() for part in code.split(" ") if part != ""]

                if parts[0].startswith("g"):

                    if parts[0] == "g0" or parts[0] == "g1":
                        for part in parts[1:]:
                            if part.startswith("x"):
                                x = float(part[1:])
                            elif part.startswith("y"):
                                y = float(part[1:])
                            elif part.startswith("z"):
                                z = float(part[1:])
                            else:
                                pass

                        x_positions.append(x)
                        y_positions.append(y)
                        z_positions.append(z)

                    else:
                        print("Skipping unknown gcode: "+" ".join(parts))

                else:
                    print("Skipping unknown command: "+" ".join(parts))

    # Should now have a list of positions, now split into pen down and pen up

    pen_up_curves = []
    pen_down_curves = []

    this_curve = []

    last_z = z_positions[0]
    curve_location = pen_down_curves if last_z < pen_touch_height else pen_up_curves

    for x, y, z in zip(x_positions, y_positions, z_positions):

        # has z crossed the touch height
        if (last_z <= pen_touch_height and z > pen_touch_height) or \
            (last_z > pen_touch_height and z <= pen_touch_height):

            this_curve.append((x, y)) # Make sure to include this point in the curve

            curve_location.append(np.array(this_curve))
            this_curve = []

            curve_location = pen_down_curves if z < pen_touch_height else pen_up_curves

        this_curve.append((x, y))

        last_z = z

    if len(this_curve) > 1:
        curve_location.append(np.array(this_curve))

    # Now plot the different types
    for curve in pen_up_curves:
        plt.plot(curve[:, 0], curve[:, 1], color='r')

    for curve in pen_down_curves:
        plt.plot(curve[:, 0], curve[:, 1], color='k')

    plt.show()
