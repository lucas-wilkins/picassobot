import xml.etree.ElementTree as et
import numpy as np

def extract_paths(filename: str):
    """ Extract paths, we can only deal with m and z commands, because who has time for more!!!!!"""

    tree = et.parse(filename)

    path_elements = [element for element in tree.iter() if element.tag.endswith("path")]

    parent_map = {c: p for p in tree.iter() for c in p}

    paths = []


    for path in path_elements:

        parent = parent_map[path]
        transform = (0, 0)
        if "transform" in parent.attrib:
            transform_string = parent.attrib["transform"]

            if transform_string.startswith("translate"):
                vector_string = transform_string.split("(")[1].split(")")[0]
                try:
                    transform = tuple([float(s) for s in vector_string.split(",")])
                except Exception:
                    print("unknown transform: "+transform_string)
            else:
                print("unknown transform: "+transform_string)

        print(transform)


        path_string = path.attrib["d"]
        print(path_string)

        parts = path_string.split(" ")

        path = []

        mode = "m"

        last_position = transform

        for part in parts:

            if part == "z":
                path.append(path[0])
                continue

            elif part.lower() in "mhvl":
                mode = part
                continue

            else:
                if mode in "lLmM":
                    # Generic line

                    try:
                        x_string, y_string = part.split(",")

                        x = float(x_string)
                        y = float(y_string)

                        if mode in "LM":
                            path.append((x, y))
                        else:
                            x0, y0 = last_position
                            path.append((x0+x, y0+y))

                    except Exception as e:
                        print(f"Unknown entry '{part}, {e}'")

                elif mode == "h" or mode == "v":
                    # Horizontal or vertical
                    try:
                        v = float(part)

                        x0, y0 = last_position

                        if mode == "h":
                            path.append((x0+v, y0))

                        elif mode == "v":
                            path.append((x0, y0+v))


                    except Exception as e:
                        print(f"Unknown entry '{part}', {e}")

            if len(path) > 0:
                last_position = path[-1]

        path = np.array(path)

        paths.append(path)


    paths = sorted(paths, key=lambda x: np.min(x[:, 0])) # sort paths left to right

    left = min([np.min(path[:, 0]) for path in paths])
    right = max([np.max(path[:, 0]) for path in paths])
    top = max([np.max(path[:, 1]) for path in paths])
    bottom = min([np.min(path[:, 1]) for path in paths])

    scale = max([right-left, top-bottom])

    for path in paths:
        path[:, 0] -= 0.5*(left+right)
        path[:, 1] -= 0.5*(top+bottom)
        path[:, 1] *= -1
        path /= scale

    import matplotlib.pyplot as plt
    for path in paths:
        plt.plot(path[:, 0], path[:, 1])
    plt.show()


    return paths



if __name__ == "__main__":

    paths = extract_paths("data/title_text.svg")

    for i, data in enumerate(paths):
        np.save(f"data/title_curve_{i}.npy", data)

    paths = extract_paths("data/makerspace_logo_cleaned.svg")

    for i, data in enumerate(paths):
        np.save(f"data/logo_curve_{i}.npy", data)

