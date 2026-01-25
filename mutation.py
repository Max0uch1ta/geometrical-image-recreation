import random
import numpy as np
from genotype import Genotype, SVGShape, Rect, Circle, Ellipse
from utils import get_random_shape 

def clip(val: int, min_val: int, max_val: int) -> int:
    """Helper to keep values within valid bounds."""
    return max(min_val, min(val, max_val))

def _get_bounds(shape: SVGShape) -> tuple[int, int, int, int]:
    """Helper to get the bounding box (x, y, w, h) for any shape type."""
    if isinstance(shape, Rect):
        return shape.x, shape.y, shape.w, shape.h
    elif isinstance(shape, Circle):
        return shape.cx - shape.r, shape.cy - shape.r, shape.r * 2, shape.r * 2
    elif isinstance(shape, Ellipse):
        return shape.cx - shape.rx, shape.cy - shape.ry, shape.rx * 2, shape.ry * 2
    return 0, 0, 1, 1


def mutate_add_shape(gen: Genotype, shape_class: type[SVGShape], original_img: np.ndarray = None, use_opacity: bool = False):
    """
    Adds a new shape using 'Smart Color' (averaging the pixels it covers) if image is provided.
    """
    new_shape = get_random_shape(shape_class, gen.w, gen.h, use_opacity=use_opacity)
    
    if original_img is not None:
        # Get bounding box
        x, y, w, h = _get_bounds(new_shape)
        
        # Clip to image boundaries to avoid crashes
        x = clip(x, 0, gen.w - 1)
        y = clip(y, 0, gen.h - 1)
        w = clip(w, 1, gen.w - x)
        h = clip(h, 1, gen.h - y)
        
        # Sample average color
        region = original_img[y:y+h, x:x+w]
        if region.size > 0:
            mean_color = region.mean(axis=(0, 1)).astype(int)
            new_shape.set_color(tuple(mean_color))
            
    gen.shapes.append(new_shape)

def mutate_resize_shape(gen: Genotype, delta_scale: float):
    """
    Incrementally changes the size of a random shape (+/- delta_scale %).
    """
    shape = random.choice(gen.shapes)
    
    if isinstance(shape, Rect):
        # Calculate mutation amount based on current size
        dw = int(random.uniform(-1, 1) * shape.w * delta_scale) + random.choice([-1, 1])
        dh = int(random.uniform(-1, 1) * shape.h * delta_scale) + random.choice([-1, 1])
        shape.set_size(clip(shape.w + dw, 5, gen.w), clip(shape.h + dh, 5, gen.h))
        
    elif isinstance(shape, Circle):
        dr = int(random.uniform(-1, 1) * shape.r * delta_scale) + random.choice([-1, 1])
        shape.set_size(clip(shape.r + dr, 3, min(gen.w, gen.h) // 2))
        
    elif isinstance(shape, Ellipse):
        drx = int(random.uniform(-1, 1) * shape.rx * delta_scale) + random.choice([-1, 1])
        dry = int(random.uniform(-1, 1) * shape.ry * delta_scale) + random.choice([-1, 1])
        shape.set_size(clip(shape.rx + drx, 3, gen.w // 2), clip(shape.ry + dry, 3, gen.h // 2))

def mutate_move_shape(gen: Genotype, move_range: int):
    """
    Incrementally moves a random shape within +/- move_range pixels.
    """
    shape = random.choice(gen.shapes)
    
    dx = random.randint(-move_range, move_range)
    dy = random.randint(-move_range, move_range)
    
    if isinstance(shape, Rect):
        shape.x = clip(shape.x + dx, -shape.w // 2, gen.w - shape.w // 2)
        shape.y = clip(shape.y + dy, -shape.h // 2, gen.h - shape.h // 2)
        
    elif isinstance(shape, Circle) or isinstance(shape, Ellipse):
        shape.cx = clip(shape.cx + dx, 0, gen.w)
        shape.cy = clip(shape.cy + dy, 0, gen.h)

def mutate_recolor_shape(gen: Genotype):
    """
    Completely randomizes the color of a shape.
    """
    shape = random.choice(gen.shapes)
    new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    shape.set_color(new_color)


def mutate_opacity_shape(gen: Genotype):
    """
    Randomizes the opacity of a shape (50%-100%).
    """
    shape = random.choice(gen.shapes)
    new_opacity = random.uniform(0.5, 1.0)
    shape.set_opacity(new_opacity)


def apply_mutation(gen: Genotype, shape_class: type[SVGShape], original_img: np.ndarray = None, 
                   mutation_type: str = "random", delta_scale: float = 0.1, move_range: int = 10,
                   use_opacity: bool = False):
    """
    Applies a mutation to the genotype in-place.
    
    Args:
        gen: The Genotype object.
        shape_class: Class of shape for 'add' mutations.
        original_img: Reference image for smart coloring (optional).
        mutation_type: Specific mutation or 'random'.
        delta_scale: Percentage (0.0-1.0) to resize shapes by.
        move_range: Max pixels to move shapes by.
        use_opacity: If True, include opacity mutations in random options.
    """
    if not gen.shapes:
        mutation_type = "add"
    
    if mutation_type == "random":
        # Include opacity mutation only if enabled
        options = ["add", "resize", "move", "recolor"]
        if use_opacity:
            options.append("opacity")
        mutation_type = random.choice(options)

    # Dispatch to specific function
    if mutation_type == "add":
        mutate_add_shape(gen, shape_class, original_img, use_opacity=use_opacity)
    elif mutation_type == "resize":
        mutate_resize_shape(gen, delta_scale)
    elif mutation_type == "move":
        mutate_move_shape(gen, move_range)
    elif mutation_type == "recolor":
        mutate_recolor_shape(gen)
    elif mutation_type == "opacity":
        mutate_opacity_shape(gen)