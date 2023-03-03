from projective_plane_visualiser import ProjectivePlaneVisualiser
from plane_objects import ProjectivePlaneDrawer, RED, BLUE, DARK_GREEN


def stand_on_plane_example(drawer):
    # draw 3 lines that help visualize the 'standing on a plane view'
    drawer.set_color(RED)
    drawer.set_line_thickness(5)
    drawer.line((0, 0, 1), (1, 0, 0))
    drawer.line((0, 0, 1), (0, 1, 0))
    drawer.set_color(DARK_GREEN)
    drawer.line((1, 0, 0), (0, 1, 0))


def stand_inside_sphere(drawer):
    # 3 lines that help visualize the 'standing inside a sphere' view
    drawer.set_color(RED)
    drawer.set_line_thickness(5)
    drawer.line((0, 0, 1), (1, 0, 0))
    drawer.line((0, 0, 1), (0, 1, 0))
    drawer.line((1, 0, 0), (0, 1, 0))
    drawer.set_line_thickness(1)
    drawer.line((0, 0, 1), (1, 1, 0))
    drawer.line((0, 0, 1), (1, -1, 0))


def elliptic_curve_example(drawer):
    # an elliptic curve
    drawer.set_color(BLUE)
    drawer.set_line_thickness(1)
    drawer.set_curve_square_size(35)
    drawer.curve(lambda x, y, z: -z*y**2+x**3-(z**2)*x+z**3)


def colored_square_example(drawer):
    # draw a nice 1x1 square (affine speaking)
    drawer.set_line_thickness(1)
    for i in range(-5, 6):
        for j in range(-5, 6):
            drawer.set_color((15*5+15*i, 15*5+15*j, 15*5-15*j))
            drawer.point(0.2*i, 0.2*j, 1)


def touching_parabolas_example(drawer):
    # two parabolas touching
    drawer.set_curve_square_size(35)
    drawer.set_color(BLUE)
    drawer.curve(lambda x, y, z: -x**2+z*y)
    drawer.set_color(RED)
    drawer.curve(lambda x, y, z: x**2+z*y)


def nice_graph_example(drawer):
    # two nice graphs where it is fun to move along them
    drawer.set_color(DARK_GREEN)
    drawer.set_line_thickness(3)
    drawer.line((0, 0, 1), (1, 0, 1))
    drawer.line((0, 0, 1), (0, 1, 1))
    drawer.set_color(BLUE)
    drawer.set_line_thickness(1)
    drawer.set_curve_square_size(35)
    drawer.curve(lambda x, y, z: x*x+z*z-x*y)
    drawer.set_color(RED)
    drawer.curve(lambda x, y, z: x*x-z*z-x*y)


def main():
    drawer = ProjectivePlaneDrawer()

    plane_distance = 10

    # stand_on_plane_example(drawer)
    # stand_inside_sphere(drawer)
    # colored_square_example(drawer)
    # elliptic_curve_example(drawer)
    # touching_parabolas_example(drawer)
    # nice_graph_example(drawer)

    visualizer = ProjectivePlaneVisualiser(drawer.objects, plane_distance=plane_distance)
    visualizer.run()


if __name__ == '__main__':
    main()
