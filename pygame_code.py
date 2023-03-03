import numpy as np
import pygame
import pygame.gfxdraw


class PygameWindow:
    def __init__(self, screen_size: tuple[int, int], screen_title: str, frame_rate: int, background_color,
                 resizable=False, scale_center=(0., 0.), tracked_keys=None):
        self._start_screen_size = screen_size
        self._screen_title = screen_title
        self.frame_rate = frame_rate
        self.background_color = background_color
        self._resizable = resizable
        self.display = None
        self._do_quit = False
        self.clock = None
        self._cur_pos = None
        self._cur_click = None
        self._delta_cur = None
        self.events = None
        self.scale_center = np.array(scale_center, dtype=float)
        self._key_tracking = dict()
        self._key_down_tracking = dict()
        self._default_font = None
        self._on_mouse_down = [False for _ in range(10)]
        self._scroll_wheel_y = 0
        if tracked_keys is not None:
            for key in tracked_keys:
                self._key_tracking[key] = False

    def __enter__(self):
        pygame.init()
        if self._resizable:
            self.display = pygame.display.set_mode(self._start_screen_size, pygame.RESIZABLE)
        else:
            self.display = pygame.display.set_mode(self._start_screen_size)
        pygame.display.set_caption(self._screen_title)
        self.clock = pygame.time.Clock()
        self.display.fill(self.background_color)
        self._cur_pos = self._screen_to_np(pygame.mouse.get_pos())
        self._cur_click = pygame.mouse.get_pressed(num_buttons=5)
        self._delta_cur = np.zeros(2, dtype=float)
        self.events = pygame.event.get()
        for key in self._key_tracking.keys():
            self._key_tracking[key] = False
            self._key_down_tracking[key] = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        pygame.quit()
        self._do_quit = False

    def do_continue(self):
        return not self._do_quit

    def next_frame(self):
        pygame.display.update()
        self.clock.tick(self.frame_rate)
        self.display.fill(self.background_color)
        last_cur_pos = self._cur_pos
        self._cur_pos = self._screen_to_np(pygame.mouse.get_pos())
        self._delta_cur = self._cur_pos-last_cur_pos
        self._cur_click = pygame.mouse.get_pressed()
        self.events = pygame.event.get()
        for key in self._key_tracking.keys():
            self._key_down_tracking[key] = False
        for i in range(len(self._on_mouse_down)):
            self._on_mouse_down[i] = False
        self._scroll_wheel_y = 0
        for event in self.events:
            if event.type == pygame.QUIT:
                self.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key in self._key_tracking:
                    self._key_tracking[event.key] = True
                    self._key_down_tracking[event.key] = True
            elif event.type == pygame.KEYUP:
                if event.key in self._key_tracking:
                    self._key_tracking[event.key] = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._on_mouse_down[event.button] = True
            elif event.type == pygame.MOUSEWHEEL:
                self._scroll_wheel_y = event.y

    def quit(self):
        self._do_quit = True

    @property
    def cur_pos(self):
        return self._cur_pos.copy()

    @property
    def delta_cur(self):
        return self._delta_cur.copy()

    def is_mouse_button_down(self, key):
        return self._cur_click[key]

    def is_key_down(self, key):
        if key not in self._key_tracking:
            raise Exception(f"Key '{key}' not tracked!")
        return self._key_tracking[key]

    def on_key_down(self, key):
        if key not in self._key_down_tracking:
            raise Exception(f"Key '{key}' not tracked!")
        return self._key_down_tracking[key]

    def get_scroll_wheel_y(self):
        return self._scroll_wheel_y

    def on_mouse_button_down(self, key):
        return self._on_mouse_down[key]

    def _np_to_screen(self, x: np.ndarray):
        x = x + np.array(self.display.get_size(), dtype=float)*self.scale_center
        return tuple(round(y) for y in x)

    def _screen_to_np(self, x: tuple[int, int]):
        return np.array(x, dtype=float)-(np.array(self.display.get_size(), dtype=float)*self.scale_center)

    @property
    def top_left(self) -> np.ndarray:
        return self._screen_to_np((0, 0))

    @property
    def bottom_left(self) -> np.ndarray:
        return self._screen_to_np((0, self.display.get_size()[1]))

    @property
    def top_right(self) -> np.ndarray:
        return self._screen_to_np((self.display.get_size()[0], 0))

    @property
    def bottom_right(self) -> np.ndarray:
        return self._screen_to_np(self.display.get_size())

    @property
    def center(self) -> np.ndarray:
        return self._screen_to_np(tuple(round(x/2) for x in self.display.get_size()))

    @property
    def size(self) -> np.ndarray:
        return np.array(self.display.get_size(), dtype=float)

    def is_in_screen(self, x):
        p = self._np_to_screen(x)
        width, height = self.display.get_size()
        return 0 <= p[0] <= width and 0 <= p[1] <= height

    def set_default_font(self, size, font_name=None, bold=False, italic=False):
        self._default_font = pygame.font.SysFont(font_name, size, bold=bold, italic=italic)

    def draw_text(self, text, color, position, offset=(0, 0), font=None, anti_alias=True):
        if font is None:
            font = self._default_font
        img = font.render(text, anti_alias, color)
        size = np.array(img.get_size(), dtype=float)
        offset = np.array(offset, dtype=float)
        pos = self._np_to_screen(position-offset*size)
        self.display.blit(img, pos)

    def draw_circle(self, center: np.ndarray, radius, color: tuple):
        r = round(radius)
        c = self._np_to_screen(center)
        if np.linalg.norm(center)-r < np.linalg.norm(self.size):
            pygame.gfxdraw.filled_circle(self.display, c[0], c[1], r, color)
            pygame.gfxdraw.aacircle(self.display, c[0], c[1], r, color)

    def draw_line(self, p1: np.ndarray, p2: np.ndarray, thickness, color):
        thickness = round(thickness)
        if thickness == 1:
            pygame.gfxdraw.line(self.display, *self._np_to_screen(p1), *self._np_to_screen(p2), color)
        elif thickness > 1:
            pygame.draw.line(self.display, color, self._np_to_screen(p1), self._np_to_screen(p2), thickness)

    def draw_rect(self, start: np.ndarray, size: np.ndarray, color, width=0):
        rect_corner = self._np_to_screen(start)
        rect_size = tuple(round(x) for x in size)
        pygame.draw.rect(self.display, color, rect_corner + rect_size, width)
