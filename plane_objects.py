from LinearSubspaceCalculator import Subspace
import abc
from dataclasses import dataclass
import numpy as np
from marching_squares import draw_zeros
from math import sqrt


class PlaneObject(abc.ABC):
    @abc.abstractmethod
    def draw(self, window, inv_transform, pixel_scale):
        pass

    @abc.abstractmethod
    def click_check(self, cur_pos, window, inv_transform, pixel_scale):
        pass

    @abc.abstractmethod
    def get_dual(self, transform, inv_transform):
        pass


def proj_point_to_screen(subspace, pixel_scale):
    a = np.concatenate((subspace.A, np.array([[0, 0, 1]], dtype=float)))
    b = np.array([0, 0, 1], dtype=float)
    try:
        return np.linalg.solve(a, b)[:-1]/pixel_scale
    except np.linalg.LinAlgError:
        return None


def screen_to_proj(p: np.ndarray, pixel_scale):
    generator = np.concatenate((p * pixel_scale, np.array([1], dtype=float)))
    return Subspace.from_generators(generator)


@dataclass
class ProjectivePoint(PlaneObject):
    subspace: Subspace
    color: tuple[int, int, int]
    radius: int
    dual_thickness: int

    def draw(self, window, inv_transform, pixel_scale):
        center = proj_point_to_screen(self.subspace.transform(inv_transform), pixel_scale)
        if center is not None:
            window.draw_circle(center, self.radius, self.color)

    def click_check(self, cur_pos, window, inv_transform, pixel_scale):
        center = proj_point_to_screen(self.subspace.transform(inv_transform), pixel_scale)
        if center is None:
            return False, None
        return np.linalg.norm(cur_pos-center) <= self.radius+1, center

    def get_dual(self, transform, inv_transform):
        new_subspace = self.subspace.transform(inv_transform).perp.transform(transform)
        return ProjectiveLine(new_subspace, self.color, self.dual_thickness, self.radius)


@dataclass
class ProjectiveLine(PlaneObject):
    subspace: Subspace
    color: tuple[int, int, int]
    thickness: int
    dual_radius: int

    def get_screen_points(self, window, inv_transform, pixel_scale):
        transformed_space = Subspace.kernel(self.subspace.A.dot(inv_transform))
        proj_corners = [
            np.concatenate((window.top_left * pixel_scale, np.array([1], dtype=float))),
            np.concatenate((window.top_right * pixel_scale, np.array([1], dtype=float))),
            np.concatenate((window.bottom_right * pixel_scale, np.array([1], dtype=float))),
            np.concatenate((window.bottom_left * pixel_scale, np.array([1], dtype=float))),
        ]
        proj_lines = [Subspace.from_generators(proj_corners[i], proj_corners[(i + 1) % 4])
                      for i in range(4)]
        found_points = []
        for line in proj_lines:
            intersect_space = Subspace.intersection(transformed_space, line)
            try:
                screen_point = proj_point_to_screen(intersect_space, pixel_scale)
            except np.linalg.LinAlgError:
                continue
            if screen_point is not None and window.is_in_screen(screen_point):
                found_points.append(screen_point)
                if len(found_points) == 2:
                    return found_points
        return None

    def draw(self, window, inv_transform, pixel_scale):
        screen_points = self.get_screen_points(window, inv_transform, pixel_scale)
        if screen_points is not None:
            window.draw_line(*screen_points, self.thickness, self.color)

    def click_check(self, cur_pos, window, inv_transform, pixel_scale):
        screen_points = self.get_screen_points(window, inv_transform, pixel_scale)
        if screen_points is None:
            return False, None
        v = normalise(screen_points[1]-screen_points[0])
        p = cur_pos-screen_points[0]
        return sqrt(np.dot(p, p)-np.dot(v, p)**2) <= self.thickness+1, screen_points[0]+v*np.dot(v, p)

    def get_dual(self, transform, inv_transform):
        new_subspace = self.subspace.transform(inv_transform).perp.transform(transform)
        return ProjectivePoint(new_subspace, self.color, self.dual_radius, self.thickness)


def normalise(v: np.ndarray):
    return v/np.linalg.norm(v)


class ProjectiveCurve(PlaneObject):
    def __init__(self, form, color, thickness, square_size):
        self.form = form
        self.color = color
        self.thickness = thickness
        self.square_size = square_size

    def draw(self, window, inv_transform, pixel_scale):
        def f(x):
            return self.form(*normalise(inv_transform.dot(np.concatenate((x*pixel_scale, np.array([1], dtype=float))))))
        draw_zeros(window, f, self.color, self.thickness, self.square_size)

    def click_check(self, cur_pos, window, inv_transform, pixel_scale):
        return False, None

    def get_dual(self, transform, inv_transform):
        return self


class DrawingException(Exception):
    pass


RED = (200, 30, 30)
BLUE = (20, 5, 255)
DARK_GREEN = (0, 75, 0)


class ProjectivePlaneDrawer:
    def __init__(self):
        self._color = (0, 0, 0)
        self._line_thickness = 1
        self._curve_square_size = 100
        self._point_radius = 5
        self.objects = []

    def set_color(self, color):
        self._color = color

    def set_line_thickness(self, thickness):
        self._line_thickness = thickness

    def set_curve_square_size(self, square_size):
        self._curve_square_size = square_size

    def set_point_radius(self, radius):
        self._point_radius = radius

    def point(self, x, y, z, hidden=False):
        subspace = Subspace.from_generators(np.array([x, y, z], dtype=float))
        p = ProjectivePoint(subspace, self._color, self._point_radius, self._line_thickness)
        if not hidden:
            self.objects.append(p)
        return p

    def line(self, p, q, hidden=False):
        if isinstance(p, tuple):
            p = self.point(*p, hidden=True)
        if isinstance(q, tuple):
            q = self.point(*q, hidden=True)
        subspace = p.subspace+q.subspace
        line = ProjectiveLine(subspace, self._color, self._line_thickness, self._point_radius)
        if not hidden:
            self.objects.append(line)
        return line

    def curve(self, form, hidden=False):
        c = ProjectiveCurve(form, self._color, self._line_thickness, self._curve_square_size)
        if not hidden:
            self.objects.append(c)
        return c

    def intersect(self, a, b, hidden=False):
        if not (isinstance(a, ProjectiveLine) and isinstance(b, ProjectiveLine)):
            raise DrawingException("Currently only taking the intersection of projective lines is supported.")
        subspace = Subspace.intersection(a, b)
        p = ProjectivePoint(subspace, self._color, self._point_radius, self._line_thickness)
        if not hidden:
            self.objects.append(p)
        return p
