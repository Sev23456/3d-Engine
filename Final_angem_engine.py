import tkinter as tk
import numpy as np
from tkinter import ttk, Menu, filedialog, messagebox
import time
from numba import njit, prange
from functools import lru_cache
import os
import struct
import re
import sys

try:
    from PIL import ImageGrab, Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@njit(parallel=True, fastmath=True)
def rotate_vertices_jit(vertices, rotation_matrix):
    n = vertices.shape[0]
    result = np.empty((n, 3))
    for i in prange(n):
        x = vertices[i, 0]
        y = vertices[i, 1]
        z = vertices[i, 2]
        result[i, 0] = rotation_matrix[0, 0] * x + rotation_matrix[0, 1] * y + rotation_matrix[0, 2] * z
        result[i, 1] = rotation_matrix[1, 0] * x + rotation_matrix[1, 1] * y + rotation_matrix[1, 2] * z
        result[i, 2] = rotation_matrix[2, 0] * x + rotation_matrix[2, 1] * y + rotation_matrix[2, 2] * z
    return result

@njit(parallel=True, fastmath=True)
def project_vertices_jit(vertices, camera_distance, fov, width, height):
    n = vertices.shape[0]
    projected = np.empty((n, 2))
    for i in prange(n):
        x = vertices[i, 0]
        y = vertices[i, 1]
        z = vertices[i, 2] + camera_distance
        
        if abs(z) > 1e-8:
            f = fov / z
            projected[i, 0] = f * x + width / 2
            projected[i, 1] = f * y + height / 2
        else:
            projected[i, 0] = width / 2
            projected[i, 1] = height / 2
    return projected

@njit(parallel=True, fastmath=True)
def calculate_face_depths_jit(rotated_vertices, faces):
    n_faces = faces.shape[0]
    depths = np.empty(n_faces)
    for i in prange(n_faces):
        face = faces[i]
        z_sum = 0.0
        count = 0
        for j in range(len(face)):
            if face[j] < rotated_vertices.shape[0]:
                z_sum += rotated_vertices[face[j], 2]
                count += 1
        depths[i] = z_sum / count if count > 0 else 0.0
    return depths

@njit(fastmath=True)
def dot_product_jit(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@njit(fastmath=True)
def normalize_jit(v):
    norm = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return v / norm

class OptimizedLambertEngine:
    def __init__(self, width=600, height=600):
        self.width = width
        self.height = height
        self.root = tk.Tk()
        self.root.title("Оптимизированный 3D Рендерер")
        
        self.max_fps = 60
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_display = False
        
        self.camera_distance = 5.0
        self.fov = 200.0
        
        self.color_presets = {
            "Синий": np.array([0.8, 0.8, 1.0]),
            "Зелёный": np.array([0.8, 1.0, 0.8]),
            "Красный": np.array([1.0, 0.8, 0.8]),
            "Жёлтый": np.array([1.0, 1.0, 0.8]),
            "Фиолетовый": np.array([1.0, 0.8, 1.0]),
            "Бирюзовый": np.array([0.8, 1.0, 1.0]),
            "Белый": np.array([1.0, 1.0, 1.0]),
            "Серый": np.array([0.7, 0.7, 0.7])
        }
        self.current_color = self.color_presets["Синий"]
        
        self.light_position = np.array([2.0, 2.0, 5.0], dtype=np.float32)
        self.ambient_light = 0.2
        
        self.sphere_detail_var = tk.IntVar(value=15)
        self.torus_detail_var = tk.IntVar(value=15)
        self.max_detail = 40
        self.min_detail = 8
        
        self.torus_R_var = tk.DoubleVar(value=1.5)
        self.torus_r_var = tk.DoubleVar(value=0.5)
        
        self.model_cache = {}
        self.loaded_models = {}
        
        self.create_menu()
        
        self.create_control_panel()
        
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.geometries = {}
        self.sphere_cache = {}
        self.torus_cache = {}
        
        self.current_primitive = "cube"
        
        self.initialize_geometries()
        
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0
        
        self.mouse_x = 0
        self.mouse_y = 0
        self.is_dragging = False
        
        self.auto_rotate = True
        
        self.render_stats = tk.StringVar(value="FPS: 0")
        stats_label = ttk.Label(self.root, textvariable=self.render_stats)
        stats_label.pack(side=tk.BOTTOM, pady=2)
        
        self.canvas.bind("<Button-1>", self.on_mouse_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.root.bind("<Configure>", self.on_resize)
        
        self.load_current_geometry()
        
        self.animate()
        
        self.last_lod_update = 0

    def create_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Загрузить STL файл", command=self.load_stl_file)
        file_menu.add_command(label="Загрузить OBJ файл", command=self.load_obj_file)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)
        
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="О программе", command=self.show_about)
        menubar.add_cascade(label="Справка", menu=help_menu)

    def create_control_panel(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Фигура:").pack(side=tk.LEFT, padx=(0, 5))
        self.primitive_var = tk.StringVar(value="cube")
        self.primitive_menu = ttk.Combobox(
            control_frame, 
            textvariable=self.primitive_var,
            values=["cube", "sphere", "torus"],
            state="readonly",
            width=10
        )
        self.primitive_menu.pack(side=tk.LEFT, padx=5)
        self.primitive_menu.bind("<<ComboboxSelected>>", lambda e: self.change_primitive(self.primitive_var.get()))
        
        ttk.Label(control_frame, text="Цвет:").pack(side=tk.LEFT, padx=(10, 5))
        self.color_var = tk.StringVar(value="Синий")
        color_menu = ttk.Combobox(
            control_frame,
            textvariable=self.color_var,
            values=list(self.color_presets.keys()),
            state="readonly",
            width=10
        )
        color_menu.pack(side=tk.LEFT, padx=5)
        color_menu.bind("<<ComboboxSelected>>", lambda e: self.change_color(self.color_var.get()))
        
        ttk.Button(control_frame, text="Сбросить", command=self.reset_rotation, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Стоп", command=self.toggle_auto_rotate, width=8).pack(side=tk.LEFT, padx=5)
        
        self.sphere_frame = ttk.Frame(control_frame)
        
        ttk.Label(self.sphere_frame, text="Детализация:").pack(side=tk.LEFT, padx=(10, 5))
        self.detail_scale = ttk.Scale(
            self.sphere_frame,
            from_=self.min_detail, to=self.max_detail,
            variable=self.sphere_detail_var,
            orient=tk.HORIZONTAL,
            length=120,
            command=self.on_detail_change
        )
        self.detail_scale.pack(side=tk.LEFT, padx=5)
        
        self.detail_label = ttk.Label(self.sphere_frame, text=f"{self.sphere_detail_var.get()}")
        self.detail_label.pack(side=tk.LEFT, padx=5)
        
        self.sphere_frame.pack_forget()
        self.sphere_controls_visible = False
        
        self.torus_frame = ttk.Frame(control_frame)
        
        ttk.Label(self.torus_frame, text="R:").pack(side=tk.LEFT, padx=(5, 2))
        R_entry = ttk.Entry(self.torus_frame, textvariable=self.torus_R_var, width=5)
        R_entry.pack(side=tk.LEFT, padx=2)
        R_entry.bind('<Return>', lambda e: self.regenerate_torus())
        
        ttk.Label(self.torus_frame, text="r:").pack(side=tk.LEFT, padx=(5, 2))
        r_entry = ttk.Entry(self.torus_frame, textvariable=self.torus_r_var, width=5)
        r_entry.pack(side=tk.LEFT, padx=2)
        r_entry.bind('<Return>', lambda e: self.regenerate_torus())
        
        ttk.Label(self.torus_frame, text="Детализация:").pack(side=tk.LEFT, padx=(10, 5))
        self.torus_detail_scale = ttk.Scale(
            self.torus_frame,
            from_=self.min_detail, to=self.max_detail,
            variable=self.torus_detail_var,
            orient=tk.HORIZONTAL,
            length=80,
            command=self.on_torus_detail_change
        )
        self.torus_detail_scale.pack(side=tk.LEFT, padx=5)
        
        self.torus_detail_label = ttk.Label(self.torus_frame, text=f"{self.torus_detail_var.get()}")
        self.torus_detail_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.torus_frame, text="Пересчитать", command=self.regenerate_torus).pack(side=tk.LEFT, padx=5)
        
        self.torus_frame.pack_forget()
        self.torus_controls_visible = False

    def initialize_geometries(self):
        cube_vertices = np.array([
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1,  1, -1], 
            [-1, -1,  1],
            [ 1, -1,  1],
            [ 1,  1,  1],
            [-1,  1,  1]
        ], dtype=np.float32)
        
        cube_faces = np.array([
            [0, 1, 2, 3],  # Передняя грань
            [4, 7, 6, 5],  # Задняя грань
            [0, 4, 5, 1],  # Нижняя грань
            [3, 2, 6, 7],  # Верхняя грань
            [0, 3, 7, 4],  # Левая грань
            [1, 5, 6, 2]   # Правая грань
        ], dtype=np.int32)
        
        cube_normals = np.array([
            [0, 0, -1],    # Передняя грань
            [0, 0, 1],     # Задняя грань
            [0, -1, 0],    # Нижняя грань
            [0, 1, 0],     # Верхняя грань
            [-1, 0, 0],    # Левая грань
            [1, 0, 0]      # Правая грань
        ], dtype=np.float32)
        
        self.geometries["cube"] = {
            'vertices': cube_vertices,
            'faces': cube_faces,
            'normals': cube_normals,
            'type': 'flat'
        }
        
        self.geometries["sphere"] = {
            'vertices': None,
            'faces': None,
            'normals': None,
            'type': 'smooth'
        }
        
        self.geometries["torus"] = {
            'vertices': None,
            'faces': None,
            'normals': None,
            'type': 'smooth'
        }

    @lru_cache(maxsize=10)
    def generate_sphere_cached(self, lat_segments, lon_segments):
        vertices = []
        faces = []
        
        total_vertices = (lat_segments + 1) * (lon_segments + 1)
        vertices = np.zeros((total_vertices, 3), dtype=np.float32)
        
        idx = 0
        for i in range(lat_segments + 1):
            theta = i * np.pi / lat_segments
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            for j in range(lon_segments + 1):
                phi = j * 2 * np.pi / lon_segments
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)
                
                vertices[idx, 0] = sin_theta * cos_phi
                vertices[idx, 1] = sin_theta * sin_phi
                vertices[idx, 2] = cos_theta
                idx += 1
        
        total_faces = lat_segments * lon_segments * 2
        faces = np.zeros((total_faces, 3), dtype=np.int32)
        
        face_idx = 0
        for i in range(lat_segments):
            for j in range(lon_segments):
                v1 = i * (lon_segments + 1) + j
                v2 = v1 + 1
                v3 = (i + 1) * (lon_segments + 1) + j
                v4 = v3 + 1
                
                # Два полигона
                faces[face_idx] = [v1, v2, v4]
                face_idx += 1
                faces[face_idx] = [v1, v4, v3]
                face_idx += 1
        
        return vertices, faces

    @lru_cache(maxsize=10)
    def generate_torus_cached(self, R, r, u_segments, v_segments):
        """
        Кэшированная генерация тора с оптимизацией
        
        R - радиус тора (расстояние от центра до центра трубки)
        r - радиус трубки
        u_segments - сегменты вокруг тора
        v_segments - сегменты вокруг трубки
        """
        total_vertices = (u_segments + 1) * (v_segments + 1)
        vertices = np.zeros((total_vertices, 3), dtype=np.float32)
        
        idx = 0
        for i in range(u_segments + 1):
            u = i * 2 * np.pi / u_segments  # Угол вокруг тора
            cos_u = np.cos(u)
            sin_u = np.sin(u)
            
            for j in range(v_segments + 1):
                v = j * 2 * np.pi / v_segments  # Угол вокруг трубки
                cos_v = np.cos(v)
                sin_v = np.sin(v)
                
                # Параметрические уравнения тора
                vertices[idx, 0] = (R + r * cos_v) * cos_u
                vertices[idx, 1] = (R + r * cos_v) * sin_u
                vertices[idx, 2] = r * sin_v
                idx += 1
        
        total_faces = u_segments * v_segments * 2
        faces = np.zeros((total_faces, 3), dtype=np.int32)
        
        face_idx = 0
        for i in range(u_segments):
            for j in range(v_segments):
                v1 = i * (v_segments + 1) + j
                v2 = v1 + 1
                v3 = (i + 1) * (v_segments + 1) + j
                v4 = v3 + 1
                
                # Два полигона
                faces[face_idx] = [v1, v2, v4]
                face_idx += 1
                faces[face_idx] = [v1, v4, v3]
                face_idx += 1
        
        return vertices, faces

    def load_current_geometry(self):
        geometry = self.geometries[self.current_primitive]
        
        if self.current_primitive == "sphere":
            detail = self.get_optimal_detail()
            key = (detail, detail * 2)
            
            if key not in self.sphere_cache:
                vertices, faces = self.generate_sphere_cached(*key)
                self.sphere_cache[key] = (vertices.copy(), faces.copy())
            
            vertices, faces = self.sphere_cache[key]
            geometry['vertices'] = vertices
            geometry['faces'] = faces
            
        elif self.current_primitive == "torus":
            R = float(self.torus_R_var.get())
            r = float(self.torus_r_var.get())
            detail = self.get_optimal_detail()
            
            key = (round(R, 2), round(r, 2), detail, detail)
            
            if key not in self.torus_cache:
                vertices, faces = self.generate_torus_cached(R, r, detail, detail)
                self.torus_cache[key] = (vertices.copy(), faces.copy())
            
            vertices, faces = self.torus_cache[key]
            geometry['vertices'] = vertices
            geometry['faces'] = faces
        
        self.vertices = geometry['vertices'].copy()
        self.faces = geometry['faces'].copy()
        self.face_normals = geometry['normals'] if geometry['normals'] is not None else None
        
        self.update_control_visibility()
        
        self.reset_rotation()

    def get_optimal_detail(self):
        current_time = time.time()
        if current_time - self.last_lod_update > 1.0:
            self.last_lod_update = current_time
            
            if self.current_primitive == "sphere":
                base_detail = self.sphere_detail_var.get()
            elif self.current_primitive == "torus":
                base_detail = self.torus_detail_var.get()
            else:
                base_detail = 20
            
            if hasattr(self, 'current_fps') and self.current_fps < 20:
                return max(self.min_detail, base_detail // 2)
            elif hasattr(self, 'current_fps') and self.current_fps > 40:
                return min(self.max_detail, base_detail + 5)
            else:
                return base_detail
        
        if self.current_primitive == "sphere":
            return self.sphere_detail_var.get()
        elif self.current_primitive == "torus":
            return self.torus_detail_var.get()
        return 20

    def change_primitive(self, value):
        if self.current_primitive != value:
            self.current_primitive = value
            self.load_current_geometry()
            self.render_frame()

    def on_detail_change(self, value):
        detail = int(float(value))
        self.sphere_detail_var.set(detail)
        self.detail_label.config(text=f"{detail}")
        
        if self.current_primitive == "sphere":
            current_detail = self.sphere_detail_var.get()
            key = (current_detail, current_detail * 2)
            if key in self.sphere_cache:
                del self.sphere_cache[key]
            
            self.load_current_geometry()
            self.render_frame()

    def on_torus_detail_change(self, value):
        detail = int(float(value))
        self.torus_detail_var.set(detail)
        self.torus_detail_label.config(text=f"{detail}")
        
        if self.current_primitive == "torus":
            R = float(self.torus_R_var.get())
            r = float(self.torus_r_var.get())
            current_detail = self.torus_detail_var.get()
            key = (round(R, 2), round(r, 2), current_detail, current_detail)
            
            if key in self.torus_cache:
                del self.torus_cache[key]
            
            self.load_current_geometry()
            self.render_frame()

    def regenerate_torus(self):
        if self.current_primitive == "torus":
            try:
                R = float(self.torus_R_var.get())
                r = float(self.torus_r_var.get())
                detail = self.torus_detail_var.get()
                
                if R <= 0 or r <= 0 or R <= r:
                    raise ValueError("Некорректные параметры тора")
                
                key = (round(R, 2), round(r, 2), detail, detail)
                
                if key in self.torus_cache:
                    del self.torus_cache[key]
                
                self.load_current_geometry()
                self.render_frame()
                
            except ValueError as e:
                self.torus_R_var.set(1.5)
                self.torus_r_var.set(0.5)
                messagebox.showerror("Ошибка", f"Некорректные параметры тора: {str(e)}")

    def update_control_visibility(self):
        if self.sphere_controls_visible:
            self.sphere_frame.pack_forget()
            self.sphere_controls_visible = False
            
        if self.torus_controls_visible:
            self.torus_frame.pack_forget()
            self.torus_controls_visible = False
        
        if self.current_primitive == "sphere":
            self.sphere_frame.pack(side=tk.LEFT, padx=10)
            self.sphere_controls_visible = True
        elif self.current_primitive == "torus":
            self.torus_frame.pack(side=tk.LEFT, padx=10)
            self.torus_controls_visible = True

    def change_color(self, color_name):
        if color_name in self.color_presets:
            self.current_color = self.color_presets[color_name]
            self.render_frame()

    def toggle_auto_rotate(self):
        self.auto_rotate = not self.auto_rotate

    def reset_rotation(self):
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0
        self.render_frame()

    def on_mouse_press(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.is_dragging = True
        self.auto_rotate = False

    def on_mouse_release(self, event):
        self.is_dragging = False
        self.root.after(3000, lambda: setattr(self, 'auto_rotate', True)) # 3 сек

    def on_mouse_drag(self, event):
        if self.is_dragging:
            dx = event.x - self.mouse_x
            dy = event.y - self.mouse_y
            
            sensitivity = 0.01
            self.angle_y += dx * sensitivity
            self.angle_x += dy * sensitivity

            self.mouse_x = event.x
            self.mouse_y = event.y
            
            self.render_frame()

    def on_resize(self, event):
        if event.widget == self.root:
            self.width = event.width
            self.height = event.height - 50
            self.canvas.config(width=self.width, height=self.height)
            self.render_frame()

    def create_rotation_matrix(self):
        rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.angle_x), -np.sin(self.angle_x)],
            [0, np.sin(self.angle_x), np.cos(self.angle_x)]
        ], dtype=np.float32)
        
        ry = np.array([
            [np.cos(self.angle_y), 0, np.sin(self.angle_y)],
            [0, 1, 0],
            [-np.sin(self.angle_y), 0, np.cos(self.angle_y)]
        ], dtype=np.float32)
        
        rz = np.array([
            [np.cos(self.angle_z), -np.sin(self.angle_z), 0],
            [np.sin(self.angle_z), np.cos(self.angle_z), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return rz @ ry @ rx

    def render_frame(self):
        start_time = time.time()
        
        rotation_matrix = self.create_rotation_matrix()
        
        rotated_vertices = rotate_vertices_jit(self.vertices, rotation_matrix)
        
        projected_vertices = project_vertices_jit(
            rotated_vertices, 
            self.camera_distance, 
            self.fov, 
            self.width, 
            self.height
        )
        
        face_depths = calculate_face_depths_jit(rotated_vertices, self.faces)
        
        sorted_indices = np.argsort(-face_depths)  # От дальних к ближним
        
        # Отрисовка
        self.canvas.delete("all")
        
        if self.current_primitive == "cube" and self.face_normals is not None:
            rotated_normals = rotate_vertices_jit(self.face_normals, rotation_matrix)
        else:
            rotated_normals = None
        
        # Оптимизированная отрисовка граней
        for face_idx in sorted_indices:
            face = self.faces[face_idx]
            
            # Получение координат вершин грани
            points = projected_vertices[face]
            
            if len(face) < 3:
                continue
                
            # Расчет нормали и освещения
            if self.current_primitive == "cube" and rotated_normals is not None:
                normal = rotated_normals[face_idx]
                face_vertices = rotated_vertices[face]
            else:
                # Для сферы и тора - нормаль как позиция вершин относительно центра
                face_vertices = rotated_vertices[face]
                face_center = np.mean(face_vertices, axis=0)
                normal = face_center - np.array([0.0, 0.0, 0.0])  # Нормаль от центра
                
                # Специальный случай для тора - более точный расчет нормали
                if self.current_primitive == "torus":
                    # Для тора нормаль направлена от центра трубки к поверхности
                    R = float(self.torus_R_var.get())
                    r = float(self.torus_r_var.get())
                    
                    # Берем первую вершину для определения положения
                    v0 = face_vertices[0]
                    # Расстояние от оси Z до проекции вершины на XY
                    dist_to_axis = np.sqrt(v0[0]**2 + v0[1]**2)
                    
                    # Нормаль тора: направлена от центра трубки к поверхности
                    if dist_to_axis > 1e-8:
                        tube_center = np.array([
                            R * v0[0] / dist_to_axis,
                            R * v0[1] / dist_to_axis,
                            0.0
                        ])
                        normal = v0 - tube_center
                    else:
                        normal = np.array([1.0, 0.0, 0.0])
            
            # Нормализация нормали
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-8:
                normal = normal / normal_norm
            else:
                normal = np.array([0.0, 0.0, 1.0])
            
            # Быстрый расчет освещения
            light_dir = self.light_position - np.mean(face_vertices, axis=0)
            light_dir_norm = np.linalg.norm(light_dir)
            if light_dir_norm > 1e-8:
                light_dir = light_dir / light_dir_norm
            
            dot_product = np.dot(normal, light_dir)
            intensity = self.ambient_light + (1.0 - self.ambient_light) * max(0.0, dot_product)
            intensity = min(1.0, max(0.0, intensity))
            
            # Расчет цвета
            color = self.current_color * intensity
            r = int(min(255, max(0, color[0] * 255)))
            g = int(min(255, max(0, color[1] * 255)))
            b = int(min(255, max(0, color[2] * 255)))
            fill_color = f'#{r:02x}{g:02x}{b:02x}'
            
            # Отрисовка грани
            coords = []
            if len(points.shape) > 1 and points.shape[1] >= 2:
                for p in points:
                    coords.extend([float(p[0]), float(p[1])])
            
            if len(coords) >= 6:  # Минимум 3 точки для полигона
                self.canvas.create_polygon(
                    coords,
                    fill=fill_color,
                    outline='',
                    width=1
                )
        
        end_time = time.time()
        frame_time = end_time - start_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        
        self.current_fps = fps
        if self.fps_display:
            self.render_stats.set(f"FPS: {fps:.1f} | Граней: {len(self.faces)} | Фигура: {self.current_primitive}")
        else:
            self.render_stats.set(f"FPS: {fps:.1f}")
        
        target_frame_time = 1.0 / self.max_fps
        if frame_time < target_frame_time:
            time.sleep(target_frame_time - frame_time)

    def animate(self):
        if self.auto_rotate:
            if hasattr(self, 'current_fps'):
                speed_factor = min(1.0, self.current_fps / 30.0)
                self.angle_x += 0.005 * speed_factor
                self.angle_y += 0.01 * speed_factor
                
                if self.current_primitive in ["sphere", "torus"]:
                    self.angle_z += 0.003 * speed_factor
        
        self.render_frame()
        self.root.after(16, self.animate)  # ~60 FPS

    def show_about(self):
        """Показать информацию о программе"""
        about_window = tk.Toplevel(self.root)
        about_window.title("О программе")
        about_window.geometry("450x280")
        
        text = f"""
Оптимизированный 3D Рендерер с освещением по Ламберту

Версия: 2.1 (с поддержкой тора и загрузки моделей)
Автор: Буцких В. В.

Поддерживаемые фигуры:
• Куб
• Сфера
• Тор (тороид)
• Загрузка STL/OBJ файлов

Ключевые оптимизации:
• JIT-компиляция критических участков (Numba)
• Векторизованные вычисления с NumPy
• Кэширование геометрии
• Динамические уровни детализации (LOD)
• Адаптивная частота кадров

Управление:
• Левая кнопка мыши: вращение
• Автоматическое вращение после 3 секунд бездействия
• Ползунки для детализации фигур
• Параметры тора: R (радиус тора), r (радиус трубки)
• Выбор цвета из пресетов
• Загрузка моделей из STL/OBJ файлов
        """
        
        label = ttk.Label(about_window, text=text, justify=tk.LEFT, font=('Arial', 9))
        label.pack(padx=10, pady=10)
        
        ttk.Button(about_window, text="OK", command=about_window.destroy).pack(pady=5)

    def load_stl_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите STL файл",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            start_time = time.time()
            model_name = os.path.splitext(os.path.basename(file_path))[0]
            
            with open(file_path, 'rb') as f:
                header = f.read(80).decode('utf-8', errors='ignore')
                is_binary = not header.startswith('solid')
            
            if is_binary:
                vertices, faces = self.load_binary_stl(file_path)
            else:
                vertices, faces = self.load_ascii_stl(file_path)
            
            vertices, faces = self.normalize_model(vertices, faces)
            
            normals = self.calculate_face_normals(vertices, faces)
            
            self.geometries[model_name] = {
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
                'type': 'smooth',
                'source': 'stl'
            }
            
            if model_name not in self.loaded_models:
                self.loaded_models[model_name] = file_path
            
            self.update_primitive_list()
            
            self.current_primitive = model_name
            self.load_current_geometry()
            
            elapsed_time = time.time() - start_time
            messagebox.showinfo("Успех", f"STL модель '{model_name}' успешно загружена!\n"
                                        f"Вершин: {len(vertices)}\n"
                                        f"Граней: {len(faces)}\n"
                                        f"Время загрузки: {elapsed_time:.2f} сек")
            
        except Exception as e:
            messagebox.showerror("Ошибка загрузки STL", f"Не удалось загрузить STL файл:\n{str(e)}")
            print(f"Ошибка загрузки STL: {e}")

    def load_binary_stl(self, file_path):
        with open(file_path, 'rb') as f:
            f.read(80)
            
            num_triangles = struct.unpack('<I', f.read(4))[0]
            
            vertices = []
            faces = []
            vertex_map = {}
            
            for i in range(num_triangles):
                # Читаем нормаль (12 байт)
                f.read(12)
                
                # Читаем 3 вершины (36 байт)
                triangle_vertices = []
                for j in range(3):
                    x, y, z = struct.unpack('<fff', f.read(12))
                    key = (round(x, 6), round(y, 6), round(z, 6))
                    
                    if key not in vertex_map:
                        vertex_map[key] = len(vertices)
                        vertices.append([x, y, z])
                    
                    triangle_vertices.append(vertex_map[key])
                
                # Читаем атрибуты (2 байта)
                f.read(2)
                
                faces.append(triangle_vertices)
        
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)

    def load_ascii_stl(self, file_path):
        vertices = []
        faces = []
        vertex_map = {}
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip().lower()
            
            if line.startswith('facet normal'):
                i += 1  # outer loop
                
                triangle_vertices = []
                for _ in range(3):
                    i += 1
                    vertex_line = lines[i].strip()
                    if vertex_line.startswith('vertex'):
                        coords = list(map(float, vertex_line.split()[1:4]))
                        key = (round(coords[0], 6), round(coords[1], 6), round(coords[2], 6))
                        
                        if key not in vertex_map:
                            vertex_map[key] = len(vertices)
                            vertices.append(coords)
                        
                        triangle_vertices.append(vertex_map[key])
                
                i += 2  # endloop и endfacet
                faces.append(triangle_vertices)
            
            i += 1
        
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)

    def load_obj_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите OBJ файл",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            start_time = time.time()
            model_name = os.path.splitext(os.path.basename(file_path))[0]
            
            vertices, normals, faces = self.parse_obj_file(file_path)
            
            if len(normals) == 0:
                normals = self.calculate_face_normals(vertices, faces)
            
            # Нормализация модели
            vertices, faces = self.normalize_model(vertices, faces)
            
            # Создание геометрии
            self.geometries[model_name] = {
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
                'type': 'smooth',
                'source': 'obj'
            }
            
            if model_name not in self.loaded_models:
                self.loaded_models[model_name] = file_path
            
            self.update_primitive_list()
            
            # Загрузка новой геометрии
            self.current_primitive = model_name
            self.load_current_geometry()
            
            elapsed_time = time.time() - start_time
            messagebox.showinfo("Успех", f"OBJ модель '{model_name}' успешно загружена!\n"
                                        f"Вершин: {len(vertices)}\n"
                                        f"Граней: {len(faces)}\n"
                                        f"Время загрузки: {elapsed_time:.2f} сек")
            
        except Exception as e:
            messagebox.showerror("Ошибка загрузки OBJ", f"Не удалось загрузить OBJ файл:\n{str(e)}")
            print(f"Ошибка загрузки OBJ: {e}")

    def parse_obj_file(self, file_path):
        vertices = []
        normals = []
        faces = []
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                if parts[0] == 'v':  # Вершина
                    vertices.append(list(map(float, parts[1:4])))
                elif parts[0] == 'vn':  # Нормаль
                    if len(parts) >= 4:
                        normals.append(list(map(float, parts[1:4])))
                elif parts[0] == 'f':  # Грань
                    face = []
                    for part in parts[1:]:
                        # Формат: v/vt/vn
                        indices = part.split('/')
                        if indices[0]:  # Проверка на пустой индекс
                            vertex_idx = int(indices[0]) - 1  # OBJ индексация с 1
                            face.append(vertex_idx)
                    
                    # Преобразование в полигоны если нужно
                    if len(face) >= 3:
                        if len(face) == 4:
                            # Четырехугольник -> два полигона
                            faces.append([face[0], face[1], face[2]])
                            faces.append([face[0], face[2], face[3]])
                        elif len(face) > 4:
                            # N-угольник -> полигоны
                            for i in range(1, len(face) - 1):
                                faces.append([face[0], face[i], face[i + 1]])
                        else:
                            faces.append(face)
        
        return (np.array(vertices, dtype=np.float32),
                np.array(normals, dtype=np.float32) if normals else np.array([]),
                np.array(faces, dtype=np.int32))

    def normalize_model(self, vertices, faces):
        if len(vertices) == 0:
            return vertices, faces
        
        center = np.mean(vertices, axis=0)
        vertices = vertices - center
        
        max_extent = np.max(np.abs(vertices))
        if max_extent > 1e-8:
            scale = 1.0 / max_extent
            vertices = vertices * scale
        
        return vertices, faces

    def calculate_face_normals(self, vertices, faces):
        normals = np.zeros((len(faces), 3), dtype=np.float32)
        
        for i, face in enumerate(faces):
            if len(face) < 3:
                continue
            
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            
            normal = np.cross(edge1, edge2)
            
            norm = np.linalg.norm(normal)
            if norm > 1e-8:
                normal = normal / norm
            
            normals[i] = normal
        
        return normals

    def update_primitive_list(self):
        primitives = ["cube", "sphere", "torus"]
        primitives.extend(list(self.loaded_models.keys()))
        
        self.primitive_var.set(self.current_primitive)
        self.primitive_menu['values'] = primitives

    def run(self):
        """Запуск приложения"""
        self.root.mainloop()

if __name__ == "__main__":
    try:
        import numba
    except ImportError:
        print("⚠️ Библиотека numba не установлена. Установите для максимальной производительности:")
        print("pip install numba numpy")
        sys.exit(1)
    
    engine = OptimizedLambertEngine()
    engine.run()