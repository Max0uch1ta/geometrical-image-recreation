from typing import Optional

class Genotype:
    """
    Container for the solution representation.
    Holds the list of shapes (DNA) and the image constraints.
    """
    def __init__(self, w: int, h: int):
        """
        Args:
            w: Image width (used as a constraint for mutations).
            h: Image height (used as a constraint for mutations).
        """
        self.w = w
        self.h = h
        self.shapes: list[SVGShape] = []
        self.fitness: float | None = None
        
    def clone(self):
        """
        Creates a deep copy of the Genotype manually (Fast).
        """
        new_geno = Genotype(self.w, self.h)
        new_geno.shapes = [shape.copy() for shape in self.shapes]
        new_geno.fitness = self.fitness
        return new_geno 



class SVGShape:
    """Base class handling the mandatory fill color."""

    def __init__(self, fill: tuple[int, int, int]):
        """
        Args:
            fill: Mandatory RGB tuple (r, g, b).
        """
        self.fill = fill

    def set_color(self, fill: tuple[int, int, int]):
        """
        Updates the shape's fill color.
        
        Args:
            fill: New RGB tuple (r, g, b).
        """
        self.fill = fill

    def _color_str(self) -> str:
        """Converts the fill tuple to 'rgb(r,g,b)' string."""
        r, g, b = self.fill
        return f"rgb({r},{g},{b})"

    def _style(self) -> str:
        """Returns the SVG style string."""
        return f'fill="{self._color_str()}"'
    
class Rect(SVGShape):
    """Represents an SVG Rectangle."""

    def __init__(self, x: int, y: int, w: int, h: int, fill: tuple[int, int, int]):
        super().__init__(fill)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def set_size(self, w: int, h: int):
        """
        Updates the rectangle dimensions safely.
        """
        if w < 1 or h < 1:
            raise ValueError("Width and Height must be positive integers.")
        self.w = w
        self.h = h

    def to_svg(self) -> str:
        return f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}" {self._style()} />'
    
    def copy(self):
        return Rect(self.x, self.y, self.w, self.h, self.fill)

class Circle(SVGShape):
    """Represents an SVG Circle."""

    def __init__(self, cx: int, cy: int, r: int, fill: tuple[int, int, int]):
        super().__init__(fill)
        self.cx = cx
        self.cy = cy
        self.r = r

    def set_size(self, r: int):
        """
        Updates the circle radius safely.
        """
        if r < 1:
            raise ValueError("Radius must be a positive integer.")
        self.r = r

    def to_svg(self) -> str:
        return f'<circle cx="{self.cx}" cy="{self.cy}" r="{self.r}" {self._style()} />'
    
    def copy(self):
        return Circle(self.cx, self.cy, self.r, self.fill)

class Ellipse(SVGShape):
    """Represents an SVG Ellipse."""

    def __init__(self, cx: int, cy: int, rx: int, ry: int, fill: tuple[int, int, int]):
        super().__init__(fill)
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry

    def set_size(self, rx: int, ry: int):
        """
        Updates the ellipse radii safely.
        """
        if rx < 1 or ry < 1:
            raise ValueError("Radii must be positive integers.")
        self.rx = rx
        self.ry = ry

    def to_svg(self) -> str:
        return f'<ellipse cx="{self.cx}" cy="{self.cy}" rx="{self.rx}" ry="{self.ry}" {self._style()} />'
    
    
    def copy(self):
        return Ellipse(self.cx, self.cy, self.rx, self.ry, self.fill)

