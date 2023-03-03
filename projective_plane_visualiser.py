import numpy as np
from math import sqrt, cos, sin, acos, pi, atan
import pygame
from pygame_code import PygameWindow
from plane_objects import ProjectivePoint, ProjectiveLine, screen_to_proj
from LinearSubspaceCalculator import Subspace
import abc


def normalise(v: np.ndarray):
    return v / np.linalg.norm(v)


def inverse_rotation(a: np.ndarray, b: np.ndarray):
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    sin_angle = sqrt(1 - cos_angle ** 2)
    rotation = np.array([
        [cos_angle, sin_angle, 0],
        [-sin_angle, cos_angle, 0],
        [0, 0, 1]
    ], dtype=float)
    e1 = normalise(a)
    e2 = normalise(b - np.dot(b, e1) * e1)
    basis_transform = np.array([e1, e2, np.cross(e1, e2)]).T
    return np.matmul(basis_transform, np.matmul(rotation, np.linalg.inv(basis_transform)))


def inverse_rotation_interpolated(a: np.ndarray, b: np.ndarray, t):
    theta = acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))*t
    cos_angle = cos(theta)
    sin_angle = sin(theta)
    rotation = np.array([
        [cos_angle, sin_angle, 0],
        [-sin_angle, cos_angle, 0],
        [0, 0, 1]
    ], dtype=float)
    e1 = normalise(a)
    e2 = normalise(b - np.dot(b, e1) * e1)
    basis_transform = np.array([e1, e2, np.cross(e1, e2)]).T
    return np.matmul(basis_transform, np.matmul(rotation, np.linalg.inv(basis_transform)))


def inverse_affine_squishing(center: np.ndarray, a: np.ndarray, b: np.ndarray):
    if np.linalg.norm(center-a) < 1e-10:
        return np.identity(3, dtype=float)
    e1 = normalise(a-center)
    e2 = np.array([-e1[1], e1[0]], dtype=float)
    Q = np.array([e1, e2], dtype=float).T
    factor = np.dot(e1, b-center)/np.dot(e1, a-center)
    scale_matrix = Q.dot(np.diag([1/factor, 1]).dot(Q.T))
    translate = center-scale_matrix.dot(center)
    final_matrix = np.zeros((3, 3), dtype=float)
    final_matrix[0:2, 0:2] = scale_matrix
    final_matrix[0:2, 2] = translate
    final_matrix[2, 2] = 1
    return final_matrix


def get_sine_t(t):
    return (1 - cos(pi*t))/2


VISUALISER_LABEL = "Projective Space Visualiser v0.1"
DEFAULT_PIXEL_SCALE = 0.005
DEFAULT_SCROLL_ZOOM_FACTOR = 1.1
DEFAULT_BUTTON_ZOOM_FACTOR = 1.5
DEFAULT_FRAME_RATE = 155
DEFAULT_SCREEN_SIZE = (1000, 700)
DEFAULT_BACKGROUND_COLOR = (0, 140, 70)
DEFAULT_TEXT_SIZE = 25
DEFAULT_TEXT_COLOR = (0, 0, 0)
DEFAULT_DRAW_COLORS = [(200, 30, 30), (255, 247, 0), (195, 67, 186), (20, 5, 255), (0, 75, 0), (0, 0, 0)]
DEFAULT_DRAW_RADIUS = 5
DEFAULT_DRAW_THICKNESS = 3


class EditorState(abc.ABC):
    def allow_switching(self):
        return True

    def on_exit(self, editor):
        pass
    
    @abc.abstractmethod
    def frame(self, editor):
        pass


def snap_pos(editor, pos, lines: bool, points: bool, excluded=None):
    if points:
        for obj in reversed(editor.objects):
            if excluded is not None and obj in excluded:
                continue
            if isinstance(obj, ProjectivePoint):
                clicked, click_pos = obj.click_check(pos, editor.window, editor.inv_transform, editor.pixel_scale)
                if clicked:
                    return click_pos
    if lines:
        for obj in reversed(editor.objects):
            if excluded is not None and obj in excluded:
                continue
            if isinstance(obj, ProjectiveLine):
                clicked, click_pos = obj.click_check(pos, editor.window, editor.inv_transform, editor.pixel_scale)
                if clicked:
                    return click_pos
    return pos


class Moving(EditorState):
    def frame(self, editor):
        # scroll_wheel zooming (just does type 1 zooming) and mouse based translation
        if not editor.window.is_key_down(pygame.K_LSHIFT):
            new_cur = editor.window.cur_pos * editor.pixel_scale
            new_cur = np.concatenate((new_cur, np.array([1], dtype=float)))
            if editor.window.is_mouse_button_down(0):
                old_cur = (editor.window.cur_pos - editor.window.delta_cur) * editor.pixel_scale
                old_cur = np.concatenate((old_cur, np.array([1], dtype=float)))
            else:
                old_cur = new_cur.copy()

            scroll_amount = editor.window.get_scroll_wheel_y()
            if scroll_amount != 0:
                editor.pixel_scale /= editor.scroll_zoom_factor**scroll_amount
                new_cur[:2] /= editor.scroll_zoom_factor**scroll_amount
            if np.linalg.norm(old_cur - new_cur) >= 1e-10:
                editor.inv_transform = editor.inv_transform.dot(inverse_rotation(old_cur, new_cur))
        else:
            zoom_factor = editor.scroll_zoom_factor ** editor.window.get_scroll_wheel_y()
            if editor.window.is_mouse_button_down(0):
                delta_cur = editor.window.delta_cur * editor.pixel_scale
            else:
                delta_cur = np.zeros(2, dtype=float)
            new_cur = editor.window.cur_pos * editor.pixel_scale
            translation = delta_cur+(1-zoom_factor)*new_cur
            matrix = np.diag(np.array([zoom_factor, zoom_factor, 1], dtype=float))
            matrix[:2, 2] = translation
            editor.inv_transform = editor.inv_transform.dot(np.linalg.inv(matrix))

        # allow resetting the transformation
        if editor.window.on_key_down(pygame.K_r):
            editor.inv_transform = np.diag(np.array([1, 1, 1 / editor.start_plane_distance], dtype=float))
            editor.pixel_scale = editor.start_pixel_scale / editor.start_plane_distance

        # allow taking the dual of all objects in the world
        if editor.window.on_key_down(pygame.K_d):
            transform = np.linalg.inv(editor.inv_transform)
            editor.objects = [obj.get_dual(transform, editor.inv_transform) for obj in editor.objects]


class AffineSquishing(EditorState):
    def __init__(self):
        self.pos1 = None
        self.pos2 = None
        self.start_inv_transform = None
    
    def frame(self, editor):
        if self.pos1 is None:
            editor.window.draw_text("Click somewhere on the screen.", editor.text_color,
                                    editor.window.top_left + np.array([10., 10.]))
            if editor.window.on_mouse_button_down(1):
                self.pos1 = editor.window.cur_pos.copy()
        elif self.pos2 is None:
            editor.window.draw_text("Now drag another point somewhere.", editor.text_color,
                                    editor.window.top_left + np.array([10., 10.]))
            editor.window.draw_line(self.pos1 + np.array([-10., 0.]),
                                    self.pos1 + np.array([10., 0.]), 3, editor.text_color)
            editor.window.draw_line(self.pos1 + np.array([0., -10.]),
                                    self.pos1 + np.array([0., 10.]), 3, editor.text_color)
            if editor.window.on_mouse_button_down(1):
                self.pos2 = editor.window.cur_pos.copy()
                self.start_inv_transform = editor.inv_transform.copy()
        else:
            editor.window.draw_text("Scaling...", editor.text_color,
                                    editor.window.top_left + np.array([10., 10.]))
            editor.window.draw_line(self.pos1 + np.array([-10., 0.]),
                                    self.pos1 + np.array([10., 0.]), 3, editor.text_color)
            editor.window.draw_line(self.pos1 + np.array([0., -10.]),
                                    self.pos1 + np.array([0., 10.]), 3, editor.text_color)
            if not editor.window.is_mouse_button_down(0):
                self.pos1 = self.pos2 = None
                return
            pos1 = self.pos1*editor.pixel_scale
            pos2 = self.pos2*editor.pixel_scale
            pos3 = editor.window.cur_pos*editor.pixel_scale
            matrix = inverse_affine_squishing(pos1, pos2, pos3)
            editor.inv_transform = self.start_inv_transform.dot(matrix)


class LineToInftySelect(EditorState):
    def frame(self, editor):
        editor.window.draw_text("Select a line to push to infinity", editor.text_color,
                                editor.window.top_left + np.array([10., 10.]))
        if editor.window.on_mouse_button_down(1):
            found_line = None
            for obj in editor.objects:
                if isinstance(obj, ProjectiveLine) and obj.click_check(editor.window.cur_pos, editor.window,
                                                                       editor.inv_transform, editor.pixel_scale)[0]:
                    found_line = obj
                    break
            if found_line is not None:
                new_state = LineToInftyAnimation()
                new_state.start_inv_transform = editor.inv_transform.copy()
                line_subspace = found_line.subspace.transform(editor.inv_transform)
                new_state.goal_normal = normalise(line_subspace.perp.get_generators()[0])
                editor.state = new_state


class DrawingState(EditorState, abc.ABC):
    @abc.abstractmethod
    def drawing_frame(self, editor):
        pass

    def frame(self, editor):
        self.drawing_frame(editor)

        n = len(editor.draw_colors)
        editor.draw_color_index = (editor.draw_color_index - editor.window.get_scroll_wheel_y()) % n
        corner = editor.window.top_left + np.array([10, 10], dtype=float)
        for i in range(n):
            if i == editor.draw_color_index:
                editor.window.draw_rect(corner-np.array([3., 3.], dtype=float), (26, 26), (255, 255, 255))
            editor.window.draw_rect(corner, (20, 20), editor.draw_colors[i])
            corner += np.array([25, 0], dtype=float)


class PointDrawing(DrawingState):
    def drawing_frame(self, editor):
        editor.window.draw_text("Click to draw points.",
                                editor.text_color, editor.window.top_left + np.array([10., 35.]))
        if editor.window.on_mouse_button_down(1):
            screen_pos = snap_pos(editor, editor.window.cur_pos, lines=True, points=False)
            subspace = screen_to_proj(screen_pos, editor.pixel_scale).transform(np.linalg.inv(editor.inv_transform))
            editor.objects.append(ProjectivePoint(subspace, editor.draw_colors[editor.draw_color_index],
                                                  editor.draw_radius, editor.draw_thickness))


class LineDrawing(DrawingState):
    def __init__(self):
        self.first_pos = None
        self.preview_line = None

    def drawing_frame(self, editor):
        editor.window.draw_text("Click to draw lines.",
                                editor.text_color, editor.window.top_left + np.array([10., 35.]))

        if self.preview_line is not None:
            subspace = screen_to_proj(editor.window.cur_pos, editor.pixel_scale)
            subspace = subspace.transform(np.linalg.inv(editor.inv_transform))
            self.preview_line.subspace = self.first_pos + subspace

        if not editor.window.on_mouse_button_down(1):
            return
        screen_pos = snap_pos(editor, editor.window.cur_pos, lines=True, points=True, excluded=[self.preview_line])
        subspace = screen_to_proj(screen_pos, editor.pixel_scale).transform(np.linalg.inv(editor.inv_transform))
        if self.first_pos is None:
            self.first_pos = subspace
            self.preview_line = ProjectiveLine(subspace, editor.draw_colors[editor.draw_color_index],
                                               editor.draw_thickness, editor.draw_radius)
            editor.objects.append(self.preview_line)
        else:
            self.preview_line.subspace = self.first_pos + subspace
            self.preview_line = None
            self.first_pos = None

    def on_exit(self, editor):
        if self.preview_line is not None:
            editor.objects.remove(self.preview_line)


class IntersectDrawing(DrawingState):
    def __init__(self):
        self.first_line = None

    def drawing_frame(self, editor):
        if self.first_line is None:
            editor.window.draw_text("Select a line.", editor.text_color,
                                    editor.window.top_left + np.array([10., 35.]))
        else:
            editor.window.draw_text("Select another line.", editor.text_color,
                                    editor.window.top_left + np.array([10., 35.]))
        if not editor.window.on_mouse_button_down(1):
            return
        found_line = None
        for obj in editor.objects:
            if isinstance(obj, ProjectiveLine) and obj.click_check(editor.window.cur_pos, editor.window,
                                                                   editor.inv_transform, editor.pixel_scale)[0]:
                found_line = obj
                break
        if found_line is None:
            return
        if self.first_line is None:
            self.first_line = found_line
        else:
            subspace = Subspace.intersection(self.first_line.subspace, found_line.subspace)
            point = ProjectivePoint(subspace, editor.draw_colors[editor.draw_color_index], editor.draw_radius,
                                    editor.draw_thickness)
            editor.objects.append(point)
            self.first_line = None


class LineToInftyAnimation(EditorState):
    def __init__(self):
        self.animation_t = 0
        self.start_inv_transform = np.identity(3, dtype=float)
        self.goal_normal = None

    def allow_switching(self):
        return False

    def frame(self, editor):
        editor.window.draw_text("Animating...", editor.text_color, editor.window.top_left + np.array([10., 10.]))
        self.animation_t += 1/editor.frame_rate
        if self.animation_t > 1:
            self.animation_t = 1
        t = get_sine_t(self.animation_t)
        rotation = inverse_rotation_interpolated(self.goal_normal, np.array([0, 0, 1], dtype=float), t)
        editor.inv_transform = self.start_inv_transform.dot(rotation)
        if self.animation_t == 1 or editor.window.on_key_down(pygame.K_ESCAPE):
            editor.state = Moving()


class Deleting(EditorState):
    def frame(self, editor):
        editor.window.draw_text("Click on an object to delete it", editor.text_color,
                                editor.window.top_left + np.array([10., 10.]))
        if not editor.window.on_mouse_button_down(1):
            return
        for obj in editor.objects.copy():
            if isinstance(obj, ProjectivePoint):
                clicked, click_pos = obj.click_check(editor.window.cur_pos, editor.window, editor.inv_transform,
                                                     editor.pixel_scale)
                if clicked:
                    editor.objects.remove(obj)
                    return
        for obj in editor.objects.copy():
            if isinstance(obj, ProjectiveLine):
                clicked, click_pos = obj.click_check(editor.window.cur_pos, editor.window, editor.inv_transform,
                                                     editor.pixel_scale)
                if clicked:
                    editor.objects.remove(obj)
                    return


EDITOR_STATE_CLASSES = {
    pygame.K_ESCAPE: Moving,
    pygame.K_y: LineToInftySelect,
    pygame.K_p: PointDrawing,
    pygame.K_l: LineDrawing,
    pygame.K_i: IntersectDrawing,
    pygame.K_DELETE: Deleting,
    pygame.K_a: AffineSquishing
}


class ProjectivePlaneVisualiser:
    def __init__(self, start_objects, start_pixel_scale=DEFAULT_PIXEL_SCALE, plane_distance=1,
                 scroll_zoom_factor=DEFAULT_SCROLL_ZOOM_FACTOR, button_zoom_factor=DEFAULT_BUTTON_ZOOM_FACTOR,
                 frame_rate=DEFAULT_FRAME_RATE, screen_size=DEFAULT_SCREEN_SIZE,
                 background_color=DEFAULT_BACKGROUND_COLOR, resizable=True, text_size=DEFAULT_TEXT_SIZE,
                 text_color=DEFAULT_TEXT_COLOR, draw_colors=None, draw_radius=DEFAULT_DRAW_RADIUS,
                 draw_thickness=DEFAULT_DRAW_THICKNESS):
        self.objects = start_objects.copy()
        self.screen_size = screen_size
        self.background_color = background_color
        self.inv_transform = np.diag(np.array([1, 1, 1 / plane_distance], dtype=float))
        self.pixel_scale = start_pixel_scale / plane_distance
        self.start_pixel_scale = start_pixel_scale
        self.start_plane_distance = plane_distance
        self.scroll_zoom_factor = scroll_zoom_factor
        self.button_zoom_factor = button_zoom_factor
        self.frame_rate = frame_rate
        self.resizable = resizable
        self.window = None
        self.text_size = text_size
        self.text_color = text_color
        self.draw_radius = draw_radius
        self.draw_thickness = draw_thickness
        if draw_colors is None:
            self.draw_colors = DEFAULT_DRAW_COLORS
        else:
            self.draw_colors = draw_colors
        self.draw_color_index = 0
        self.state = Moving()

    def run(self):
        tracked_keys = [
            pygame.K_r,
            pygame.K_d,
            pygame.K_LSHIFT
        ] + [key for key in EDITOR_STATE_CLASSES.keys()]
        self.window = PygameWindow(self.screen_size, VISUALISER_LABEL, self.frame_rate, self.background_color,
                                   resizable=self.resizable, scale_center=(0.5, 0.5), tracked_keys=tracked_keys)
        with self.window:
            self.window.set_default_font(self.text_size)
            while self.window.do_continue():
                for obj in self.objects:
                    obj.draw(self.window, self.inv_transform, self.pixel_scale)
                if self.state.allow_switching():
                    for key, value in EDITOR_STATE_CLASSES.items():
                        if self.window.is_key_down(key):
                            self.state.on_exit(self)
                            self.state = value()
                self.state.frame(self)
                self.draw_corner_text()
                self.window.next_frame()

    def draw_corner_text(self):
        b = self.window.cur_pos * self.pixel_scale
        b = np.concatenate((b, np.array([1], dtype=float)))
        b = self.inv_transform.dot(b)
        b = b / np.linalg.norm(b)
        if abs(b[2]) >= 1e-3:
            b /= b[-1]
            corner_text = f"cur_pos: [{b[0]:.2f} : {b[1]:.2f} : 1]"
        elif abs(b[1]) >= 1e-3:
            b /= b[1]
            corner_text = f"cur_pos: ~[{b[0]:.2f} : 1 : 0]"
        else:
            corner_text = "cur_pos: ~[1 : 0 : 0]"
        self.window.draw_text(corner_text, self.text_color, self.window.bottom_left, (0, 1))
        fov_text = f"fov: {2*atan(self.window.size[0]*self.pixel_scale/2):.2f}"
        self.window.draw_text(fov_text, self.text_color, self.window.bottom_left + np.array([0., -25.]), (0, 1))

        self.window.draw_text(f"fps: {self.window.clock.get_fps():.1f}", self.text_color, self.window.bottom_right,
                              (1, 1))
