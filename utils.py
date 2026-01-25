import numpy as np
import random
from genotype import Genotype, SVGShape, Rect, Circle, Ellipse
from tqdm import tqdm
from PIL import Image, ImageDraw

def load_img(image_path: str, max_size=512):
    img = Image.open(image_path)
    
    # Resize while maintaining aspect ratio
    img.thumbnail((max_size, max_size))
    
    # Ensure we are in RGB mode (removes Alpha channel if present, which simplifies math)
    img = img.convert("RGB")
    
    # Creates an array: [height, width, channels]
    grid = np.array(img)
    w = img.width
    h = img.height
    
    print(f"Image loaded and resized to: {w}x{h}")
    return (grid, w, h)


def get_random_shape(shape_class: type[SVGShape], w_img: int, h_img: int, min_size: int = 25, use_opacity: bool = False) -> SVGShape:
    """
    Generates a random shape (Rect, Circle, Ellipse) with randomized size, position, and optionally opacity.
    """
    # Random color (R, G, B)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # Random opacity between 50% and 100% (only if enabled)
    opacity = random.uniform(0.5, 1.0) if use_opacity else 1.0
    
    if shape_class == Rect:
        w = random.randint(min_size, w_img)
        h = random.randint(min_size, h_img)
        # Ensure x,y are within bounds
        x = random.randint(0, max(0, w_img - w))
        y = random.randint(0, max(0, h_img - h))
        return Rect(x, y, w, h, fill=color, opacity=opacity)
        
    elif shape_class == Circle:
        max_r = min(w_img, h_img) // 2
        r = random.randint(min_size // 2, max(min_size // 2 + 1, max_r))
        cx = random.randint(r, w_img - r)
        cy = random.randint(r, h_img - r)
        return Circle(cx, cy, r, fill=color, opacity=opacity)
        
    elif shape_class == Ellipse:
        max_rx = w_img // 2
        max_ry = h_img // 2
        rx = random.randint(min_size // 2, max(min_size // 2 + 1, max_rx))
        ry = random.randint(min_size // 2, max(min_size // 2 + 1, max_ry))
        cx = random.randint(rx, w_img - rx)
        cy = random.randint(ry, h_img - ry)
        return Ellipse(cx, cy, rx, ry, fill=color, opacity=opacity)

    else:
        raise ValueError(f"Unknown shape type: {shape_class}")

def apply_mutation(gen: Genotype, shape_class: type[SVGShape], mutation_type: str = "random"):
    """
    Applies a mutation to the genotype in-place.
    
    Args:
        gen: The Genotype object to modify.
        shape_class: The class of shape (Rect, Circle, etc.) to use if adding a new shape.
        mutation_type: The specific operation to perform ('add', 'resize', 'recolor'). 
                       If 'random', one is chosen automatically.
    """
    
    # 1. Determine Mutation Type
    # If the genotype is empty, we must add a shape.
    if not gen.shapes:
        mutation_type = "add"
    elif mutation_type == "random":
        # Choose a random mutation from the available options
        options = ["add", "resize", "recolor"]
        # options.append("remove") # Uncomment to enable shape removal
        mutation_type = random.choice(options)

    
    ############################### MUTATION SWITCH ###################################


    # CASE 1: ADD A NEW SHAPE
    if mutation_type == "add":
        # Use the constraints stored in the genotype (gen.w, gen.h)
        new_shape = get_random_shape(shape_class, gen.w, gen.h)
        gen.shapes.append(new_shape)

    # CASE 2: RECOLOR AN EXISTING SHAPE
    elif mutation_type == "recolor":
        shape = random.choice(gen.shapes)
        # Generate a new random RGB color
        new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        shape.set_color(new_color)

    # CASE 3: RESIZE AN EXISTING SHAPE
    elif mutation_type == "resize":
        shape = random.choice(gen.shapes)
        
        # We must check the instance type to call the correct set_size method
        if isinstance(shape, Rect):
            # Randomize width and height within image bounds
            new_w = random.randint(25, gen.w)
            new_h = random.randint(25, gen.h)
            shape.set_size(new_w, new_h)
            
        elif isinstance(shape, Circle):
            # Randomize radius
            max_r = min(gen.w, gen.h) // 2
            new_r = random.randint(12, max_r)
            shape.set_size(new_r)
            
        elif isinstance(shape, Ellipse):
            # Randomize both radii
            max_rx = gen.w // 2
            max_ry = gen.h // 2
            new_rx = random.randint(12, max_rx)
            new_ry = random.randint(12, max_ry)
            shape.set_size(new_rx, new_ry)

    # CASE 4: REMOVE A SHAPE
    elif mutation_type == "remove" and len(gen.shapes) > 0:
        index = random.randint(0, len(gen.shapes) - 1)
        gen.shapes.pop(index)



def initialize_grid(original_img: np.ndarray, w: int, h: int, shape_class: type[SVGShape], grid_size: int = 5) -> Genotype:
    """
    Creates a Genotype with a grid of shapes approximating the image.
    Each shape takes the average color of its grid cell.
    """
    geno = Genotype(w, h)
    
    # Calculate cell dimensions
    cell_w = w // grid_size
    cell_h = h // grid_size
    
    for row in range(grid_size):
        for col in range(grid_size):
            # Define cell coord
            x_start = col * cell_w
            y_start = row * cell_h
            
            # Handle edge cases for the last column/row to ensure full coverage
            x_end = (col + 1) * cell_w if col < grid_size - 1 else w
            y_end = (row + 1) * cell_h if row < grid_size - 1 else h
            
            # 2. Extract Region & Compute Mean Color
            # numpy slicing: [y:y_end, x:x_end]
            region = original_img[y_start:y_end, x_start:x_end]
            
            # Calculate average R, G, B for this slice (axis=(0,1) averages over height and width)
            if region.size == 0: continue
            mean_color = region.mean(axis=(0, 1)).astype(int)
            color_tuple = tuple(mean_color)
            
            # 3. Create Shape based on class
            if shape_class == Rect:
                # Rectangle fills the cell exactly
                shape = Rect(x_start, y_start, x_end - x_start, y_end - y_start, color_tuple)
            
            elif shape_class == Circle:
                # Circle centered in cell
                curr_w = x_end - x_start
                curr_h = y_end - y_start
                cx = x_start + curr_w // 2
                cy = y_start + curr_h // 2
                r = min(curr_w, curr_h) // 2
                shape = Circle(cx, cy, r, color_tuple)
                
            elif shape_class == Ellipse:
                # Ellipse centered in cell
                curr_w = x_end - x_start
                curr_h = y_end - y_start
                cx = x_start + curr_w // 2
                cy = y_start + curr_h // 2
                rx = curr_w // 2
                ry = curr_h // 2
                shape = Ellipse(cx, cy, rx, ry, color_tuple)

            geno.shapes.append(shape)
            
    return geno


def get_jittered_grid_individual(w: int, h: int, shape_class: type[SVGShape], grid_size: int = 5, 
                                  jitter_amount: int = 5, original_img: np.ndarray = None,
                                  use_opacity: bool = False) -> Genotype:
    """
    Creates an individual with shapes arranged in a jittered grid.
    Samples a random pixel from within the grid cell to determine its color.
    This provides a heuristic starting point for genetic algorithms.
    """
    def clip(val, min_val, max_val):
        return max(min_val, min(val, max_val))
    
    geno = Genotype(w, h)
    
    cell_w = w // grid_size
    cell_h = h // grid_size
    
    for r in range(grid_size):
        for c in range(grid_size):
            # Grid cell boundaries
            cell_x_start = c * cell_w
            cell_y_start = r * cell_h
            cell_x_end = min((c + 1) * cell_w, w - 1)
            cell_y_end = min((r + 1) * cell_h, h - 1)
            
            # Add jitter to position
            jx = random.randint(-jitter_amount, jitter_amount)
            jy = random.randint(-jitter_amount, jitter_amount)
            
            # Random opacity (50%-100%) if enabled
            opacity = random.uniform(0.5, 1.0) if use_opacity else 1.0
            
            # Create the shape based on type
            if shape_class == Rect:
                final_x = clip(cell_x_start + jx, 0, w - 1)
                final_y = clip(cell_y_start + jy, 0, h - 1)
                final_w = clip(cell_w + random.randint(-5, 5), 1, w - final_x)
                final_h = clip(cell_h + random.randint(-5, 5), 1, h - final_y)
                shape = Rect(final_x, final_y, final_w, final_h, (128, 128, 128), opacity)
                
            elif shape_class == Circle:
                cx = clip(cell_x_start + (cell_w // 2) + jx, 0, w)
                cy = clip(cell_y_start + (cell_h // 2) + jy, 0, h)
                r_size = random.randint(5, max(5, cell_w // 2))
                shape = Circle(cx, cy, r_size, (128, 128, 128), opacity)
                
            elif shape_class == Ellipse:
                cx = clip(cell_x_start + (cell_w // 2) + jx, 0, w)
                cy = clip(cell_y_start + (cell_h // 2) + jy, 0, h)
                rx = random.randint(5, max(5, cell_w // 2))
                ry = random.randint(5, max(5, cell_h // 2))
                shape = Ellipse(cx, cy, rx, ry, (128, 128, 128), opacity)

            # Sample a RANDOM pixel from the grid cell
            if original_img is not None:
                sample_x = random.randint(cell_x_start, cell_x_end)
                sample_y = random.randint(cell_y_start, cell_y_end)
                pixel_color = original_img[sample_y, sample_x]
                shape.set_color(tuple(pixel_color.astype(int)))
            
            geno.shapes.append(shape)
            
    return geno


def local_search(geno: Genotype, original_img: np.ndarray, shape_class: type[SVGShape], max_it: int = 1000, patience: int = 200) -> Genotype:
    
    best_geno = geno.clone()
    best_phenotype = generate_phenotype(best_geno.w, best_geno.h, best_geno.shapes)
    best_score = compute_fitness(original_img, best_phenotype)
    
    # Stagnation Counter
    no_improve_counter = 0
    
    pbar = tqdm(range(max_it), desc="Optimizing")
    
    for i in pbar:
        neighbour = best_geno.clone()
        apply_mutation(neighbour, shape_class=shape_class, mutation_type="random")
        
        neighbour_phenotype = generate_phenotype(neighbour.w, neighbour.h, neighbour.shapes)
        neighbour_score = compute_fitness(original_img, neighbour_phenotype)
        
        if neighbour_score < best_score:
            best_geno = neighbour
            best_score = neighbour_score
            
            # RESET counter because we made progress
            no_improve_counter = 0
            
            pbar.set_postfix({"Score": f"{best_score:,}", "Stag": 0})
        else:
            # INCREMENT counter
            no_improve_counter += 1
            pbar.set_postfix({"Score": f"{best_score:,}", "Stag": no_improve_counter})
            
        # STOPPING CONDITION
        if no_improve_counter >= patience:
            print(f"Stopping early due to stagnation at iteration {i}")
            break
    
    return best_geno


def generate_phenotype(width, height, genotype):
    """
    Renders shapes onto a canvas with proper alpha compositing.
    """
    # Create a blank RGBA canvas (black background, fully opaque)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))

    for shape in genotype:
        # Create a temporary RGBA layer for this shape
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        
        # Convert opacity (0.0-1.0) to alpha (0-255)
        alpha = int(shape.opacity * 255)
        fill_rgba = (*shape.fill, alpha)
        
        if isinstance(shape, Rect):
            x1 = shape.x + shape.w
            y1 = shape.y + shape.h
            draw.rectangle([shape.x, shape.y, x1, y1], fill=fill_rgba)
            
        elif isinstance(shape, Circle):
            x0 = shape.cx - shape.r
            y0 = shape.cy - shape.r
            x1 = shape.cx + shape.r
            y1 = shape.cy + shape.r
            draw.ellipse([x0, y0, x1, y1], fill=fill_rgba)

        elif isinstance(shape, Ellipse):
            x0 = shape.cx - shape.rx
            y0 = shape.cy - shape.ry
            x1 = shape.cx + shape.rx
            y1 = shape.cy + shape.ry
            draw.ellipse([x0, y0, x1, y1], fill=fill_rgba)
        
        # Alpha composite this layer onto the canvas
        canvas = Image.alpha_composite(canvas, layer)

    # Convert back to RGB (discard alpha) for fitness calculation
    return np.array(canvas.convert("RGB"))


# Calculate distance between phenotype and original image
def compute_fitness(target, candidate):
    """
    L2 Norm (Sum of Squared Differences).
    Penalizes large errors more aggressively.
    """
    # Cast to int64 because of squares
    diff = target.astype(np.int64) - candidate.astype(np.int64)
    
    # Square the differences and sum them up (L2 Norm penalizes large diff)
    return np.sum(diff ** 2)


def compute_fitness_l1(target, candidate):
    """
    L1 Norm (Manhattan Distance).
    Penalizes errors linearly. Less aggressive on outliers.
    """
    # Still need int64 to avoid overflow during subtraction
    diff = target.astype(np.int64) - candidate.astype(np.int64)
    
    return np.sum(np.abs(diff))