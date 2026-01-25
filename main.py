"""
Main entry point for geometric image recreation optimization.

Usage:
    uv run python main.py --algorithm hill_climbing --max-iter 100
    uv run python main.py -a ga_tournament -n 50 --track-by-time
"""

import argparse

from genotype import Rect, Circle, Ellipse, Triangle
from utils import load_img, initialize_grid
from algorithms import (
    hill_climbing,
    simulated_annealing,
    ga_greedy,
    ga_tournament,
)


def print_history(history: list[tuple[float, float]], track_by_time: bool, sample_rate: int = 10):
    """Print fitness history at regular intervals."""
    label = "Time (s)" if track_by_time else "Iteration"
    print(f"\n{'=' * 50}")
    print(f"Fitness History (sampled every {sample_rate} entries):")
    print(f"{'=' * 50}")
    print(f"{label:>12} | {'Fitness':>15}")
    print("-" * 30)
    for i, (x, fitness) in enumerate(history):
        if i % sample_rate == 0 or i == len(history) - 1:
            if track_by_time:
                print(f"{x:>12.2f} | {fitness:>15,.0f}")
            else:
                print(f"{int(x):>12} | {fitness:>15,.0f}")
    print(f"{'=' * 50}\n")


def plot_fitness(loss_history: list[tuple[float, float]], track_by_time: bool, filename: str = None):
    """Plot fitness history using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is not installed. Cannot plot fitness history.")
        return

    x_values = [h[0] for h in loss_history]
    y_values = [h[1] for h in loss_history]
    
    label = "Time (s)" if track_by_time else "Iteration"
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values)
    plt.title("Fitness Evolution (Lower is Better)")
    plt.xlabel(label)
    plt.ylabel("Loss (SSD/L1)")
    plt.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(filename)
        print(f"Fitness plot saved to: {filename}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plot ({e}). Saving enabled if output directory is set.")


def main():
    parser = argparse.ArgumentParser(
        description="Optimization algorithms for geometric image recreation"
    )
    parser.add_argument(
        "--image", "-i", type=str, default="./test.png",
        help="Path to the target image"
    )
    parser.add_argument(
        "--algorithm", "-a", type=str, default="hill_climbing",
        choices=["hill_climbing", "simulated_annealing", "ga_greedy", "ga_tournament"],
        help="Algorithm to use"
    )
    parser.add_argument(
        "--max-iter", "-n", type=int, default=100,
        help="Maximum iterations/generations"
    )
    parser.add_argument(
        "--shape", "-s", type=str, default="rect",
        choices=["rect", "circle", "ellipse", "triangle"],
        help="Shape type to use"
    )
    parser.add_argument(
        "--track-by-time", "-t", action="store_true",
        help="Track fitness by elapsed seconds instead of iterations"
    )
    parser.add_argument(
        "--pop-size", "-p", type=int, default=50,
        help="Population size (for genetic algorithms)"
    )
    parser.add_argument(
        "--grid-init", "-g", type=int, default=5,
        help="Grid size for initial seeding (for local search)"
    )
    parser.add_argument(
        "--use-opacity", "-o", action="store_true",
        help="Enable random opacity (50%%-100%%) for shapes"
    )
    parser.add_argument(
        "--fitness", "-f", type=str, default="l2",
        choices=["l1", "l2"],
        help="Fitness metric: l1 (Manhattan) or l2 (SSD, default)"
    )
    parser.add_argument(
        "--heuristic", "-H", action="store_true",
        help="Use heuristic grid initialization for GA (colors sampled from image)"
    )
    parser.add_argument(
        "--max-time", "-T", type=float, default=None,
        help="Maximum time in minutes (overrides --max-iter when set)"
    )
    parser.add_argument(
        "--show-loss", action="store_true",
        help="Plot fitness history using matplotlib"
    )
    parser.add_argument(
        "--print-history", action="store_true",
        help="Print tabular fitness history to console"
    )
    
    args = parser.parse_args()
    
    # Map shape string to class
    shape_map = {"rect": Rect, "circle": Circle, "ellipse": Ellipse, "triangle": Triangle}
    shape_class = shape_map[args.shape]
    
    # Map fitness string to function
    from utils import compute_fitness, compute_fitness_l1
    fitness_map = {"l1": compute_fitness_l1, "l2": compute_fitness}
    fitness_fn = fitness_map[args.fitness]
    
    # Convert max_time from minutes to seconds
    max_time_sec = args.max_time * 60 if args.max_time is not None else None
    
    # Load image
    original_img, w, h = load_img(args.image)
    
    # Run selected algorithm
    if args.algorithm == "hill_climbing":
        start_geno = initialize_grid(original_img, w, h, shape_class, grid_size=args.grid_init)
        best_geno, history = hill_climbing(
            start_geno, original_img, shape_class,
            max_it=args.max_iter, track_by_time=args.track_by_time,
            use_opacity=args.use_opacity, fitness_fn=fitness_fn,
            max_time=max_time_sec
        )
        
    elif args.algorithm == "simulated_annealing":
        start_geno = initialize_grid(original_img, w, h, shape_class, grid_size=args.grid_init)
        best_geno, history = simulated_annealing(
            start_geno, original_img, shape_class,
            max_it=args.max_iter, track_by_time=args.track_by_time,
            use_opacity=args.use_opacity, fitness_fn=fitness_fn,
            max_time=max_time_sec
        )
        
    elif args.algorithm == "ga_greedy":
        best_geno, history = ga_greedy(
            original_img, shape_class,
            pop_size=args.pop_size, max_it=args.max_iter,
            track_by_time=args.track_by_time,
            use_opacity=args.use_opacity, fitness_fn=fitness_fn,
            use_heuristic=args.heuristic,
            max_time=max_time_sec
        )
        
    elif args.algorithm == "ga_tournament":
        best_geno, history = ga_tournament(
            original_img, shape_class,
            pop_size=args.pop_size, max_it=args.max_iter,
            track_by_time=args.track_by_time,
            use_opacity=args.use_opacity, fitness_fn=fitness_fn,
            use_heuristic=args.heuristic,
            max_time=max_time_sec
        )
    
    # Results and Output Handling
    print(f"Final fitness: {history[-1][1]:,.0f}")
    print(f"Number of shapes: {len(best_geno.shapes)}")
    
    from datetime import datetime
    from pathlib import Path
    
    # Common elements for filenames
    image_name = Path(args.image).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"{image_name}_{args.algorithm}_{timestamp}"
    
    # Ensure output directory exists
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    # Save SVG
    save_svg(best_geno, output_dir, output_base)
    
    # Print results
    if args.print_history:
        print_history(history, args.track_by_time)
    
    if args.show_loss:
        plot_path = output_dir / f"{output_base}_loss.png"
        plot_fitness(history, args.track_by_time, str(plot_path))


def save_svg(geno, output_dir, output_base):
    """
    Saves the genotype as an SVG file.
    """
    output_path = output_dir / f"{output_base}.svg"
    
    # Generate SVG content
    svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{geno.w}" height="{geno.h}">'
    svg_background = f'<rect width="100%" height="100%" fill="black"/>'
    svg_shapes = "\n".join(shape.to_svg() for shape in geno.shapes)
    svg_footer = "</svg>"
    
    svg_content = f"{svg_header}\n{svg_background}\n{svg_shapes}\n{svg_footer}"
    
    # Write to file
    with open(output_path, "w") as f:
        f.write(svg_content)
    
    print(f"SVG saved to: {output_path}")


if __name__ == "__main__":
    main()
