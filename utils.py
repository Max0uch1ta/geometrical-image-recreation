import random
from genotype import Genotype, SVGShape, Rect, Circle, Ellipse

def get_random_shape(shape_class: type[SVGShape], w_img: int, h_img: int, min_size: int = 25) -> SVGShape:
    """
    Generates a random shape (Rect, Circle, Ellipse) with randomized size and position.
    """
    # Random color (R, G, B)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    if shape_class == Rect:
        w = random.randint(min_size, w_img)
        h = random.randint(min_size, h_img)
        # Ensure x,y are within bounds
        x = random.randint(0, max(0, w_img - w))
        y = random.randint(0, max(0, h_img - h))
        return Rect(x, y, w, h, fill=color)
        
    elif shape_class == Circle:
        max_r = min(w_img, h_img) // 2
        r = random.randint(min_size // 2, max(min_size // 2 + 1, max_r))
        cx = random.randint(r, w_img - r)
        cy = random.randint(r, h_img - r)
        return Circle(cx, cy, r, fill=color)
        
    elif shape_class == Ellipse:
        max_rx = w_img // 2
        max_ry = h_img // 2
        rx = random.randint(min_size // 2, max(min_size // 2 + 1, max_rx))
        ry = random.randint(min_size // 2, max(min_size // 2 + 1, max_ry))
        cx = random.randint(rx, w_img - rx)
        cy = random.randint(ry, h_img - ry)
        return Ellipse(cx, cy, rx, ry, fill=color)

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