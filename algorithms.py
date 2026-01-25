"""
Optimization Algorithms for Geometric Image Recreation

Contains:
- Hill Climbing (local search)
- Simulated Annealing (local search)  
- Genetic Algorithm with Greedy selection
- Genetic Algorithm with Tournament selection

All algorithms support fitness tracking per iteration or per second via --track-by-time flag.
"""

import math
import random
import time
from typing import Literal

import numpy as np
from utils import (
    load_img,
    get_random_shape,
    initialize_grid,
    generate_phenotype,
    compute_fitness,
    compute_fitness_l1,
    get_jittered_grid_individual,
    tqdm,  # Import robust tqdm
)


# =============================================================================
# Local Search Algorithms
# =============================================================================

def hill_climbing(
    geno: Genotype,
    original_img: np.ndarray,
    shape_class: type[SVGShape],
    max_it: int = 1000,
    patience: int = 1000,
    track_by_time: bool = False,
    use_opacity: bool = False,
    fitness_fn=None,
    max_time: float = None,
) -> tuple[Genotype, list[tuple[float, float]]]:
    """
    Hill Climbing local search - only accepts improving neighbors.
    
    Args:
        geno: Initial genotype
        original_img: Target image as numpy array
        shape_class: Shape class to use (Rect, Circle, Ellipse)
        max_it: Maximum iterations
        patience: Stop after this many iterations without improvement
        track_by_time: If True, track fitness by elapsed seconds; otherwise by iteration
        use_opacity: If True, enable random opacity mutations
        fitness_fn: Fitness function to use (default: compute_fitness L2)
        max_time: Maximum time in seconds (None = no time limit)
        
    Returns:
        Tuple of (best_genotype, history) where history is list of (iteration_or_time, fitness)
    """
    if fitness_fn is None:
        fitness_fn = compute_fitness
        
    best_geno = geno.clone()
    best_phenotype = generate_phenotype(best_geno.w, best_geno.h, best_geno.shapes)
    best_score = fitness_fn(original_img, best_phenotype)
    
    no_improve_counter = 0
    history: list[tuple[float, float]] = []
    start_time = time.time()
    
    pbar = tqdm(range(max_it), desc="Hill Climbing")
    
    for i in pbar:
        # Check time limit
        elapsed = time.time() - start_time
        if max_time is not None and elapsed >= max_time:
            print(f"Stopping at iteration {i} - time limit reached ({max_time:.1f}s)")
            break
            
        neighbour = best_geno.clone()
        apply_mutation(neighbour, shape_class=shape_class, original_img=original_img, use_opacity=use_opacity)
        
        neighbour_phenotype = generate_phenotype(neighbour.w, neighbour.h, neighbour.shapes)
        neighbour_score = fitness_fn(original_img, neighbour_phenotype)
        
        if neighbour_score < best_score:
            best_geno = neighbour
            best_score = neighbour_score
            no_improve_counter = 0
        else:
            no_improve_counter += 1
        
        # Record history
        if track_by_time:
            history.append((time.time() - start_time, best_score))
        else:
            history.append((float(i), best_score))
        
        pbar.set_postfix({"Score": f"{best_score:,}", "Stag": no_improve_counter})
        
        if no_improve_counter >= patience:
            print(f"Stopping early due to stagnation at iteration {i}")
            break
    
    return best_geno, history


def simulated_annealing(
    geno: Genotype,
    original_img: np.ndarray,
    shape_class: type[SVGShape],
    max_it: int = 5000,
    initial_temp: float = 10000.0,
    cooling_rate: float = 0.995,
    track_by_time: bool = False,
    use_opacity: bool = False,
    fitness_fn=None,
    max_time: float = None,
) -> tuple[Genotype, list[tuple[float, float]]]:
    """
    Simulated Annealing - probabilistically accepts worse solutions based on temperature.
    
    Args:
        geno: Initial genotype
        original_img: Target image as numpy array
        shape_class: Shape class to use (Rect, Circle, Ellipse)
        max_it: Maximum iterations
        initial_temp: Starting temperature
        cooling_rate: Temperature multiplier per iteration (0 < rate < 1)
        track_by_time: If True, track fitness by elapsed seconds; otherwise by iteration
        use_opacity: If True, enable random opacity mutations
        fitness_fn: Fitness function to use (default: compute_fitness L2)
        max_time: Maximum time in seconds (None = no time limit)
        
    Returns:
        Tuple of (best_genotype, history) where history is list of (iteration_or_time, fitness)
    """
    if fitness_fn is None:
        fitness_fn = compute_fitness
        
    current_geno = geno.clone()
    current_phenotype = generate_phenotype(current_geno.w, current_geno.h, current_geno.shapes)
    current_score = fitness_fn(original_img, current_phenotype)
    
    best_geno = current_geno.clone()
    best_score = current_score
    
    T = initial_temp
    history: list[tuple[float, float]] = []
    start_time = time.time()
    
    pbar = tqdm(range(max_it), desc="Simulated Annealing")
    
    for i in pbar:
        # Check time limit
        elapsed = time.time() - start_time
        if max_time is not None and elapsed >= max_time:
            print(f"Stopping at iteration {i} - time limit reached ({max_time:.1f}s)")
            break
            
        neighbour = current_geno.clone()
        apply_mutation(neighbour, shape_class=shape_class, original_img=original_img, use_opacity=use_opacity)
        
        neighbour_phenotype = generate_phenotype(neighbour.w, neighbour.h, neighbour.shapes)
        neighbour_score = fitness_fn(original_img, neighbour_phenotype)
        
        dif = neighbour_score - current_score
        accept = False
        
        if dif < 0:
            accept = True
        else:
            # Accept worse solution with probability based on temperature
            if T > 0.001:
                probability = math.exp(-dif / T)
                if random.random() < probability:
                    accept = True
        
        if accept:
            current_geno = neighbour
            current_score = neighbour_score
            
            if current_score < best_score:
                best_geno = current_geno.clone()
                best_score = current_score
        
        # Cool down
        T *= cooling_rate
        
        # Record history
        if track_by_time:
            history.append((time.time() - start_time, best_score))
        else:
            history.append((float(i), best_score))
        
        pbar.set_postfix({"Best": f"{best_score:,}", "Temp": f"{T:.1f}"})
    
    return best_geno, history


# =============================================================================
# Genetic Algorithm Helpers
# =============================================================================

def crossover_uniform(parent1: Genotype, parent2: Genotype) -> Genotype:
    """
    Standard crossover: Child gets 50% of genes from each parent.
    """
    child = Genotype(parent1.w, parent1.h)
    min_len = min(len(parent1.shapes), len(parent2.shapes))
    
    for i in range(min_len):
        if random.random() > 0.5:
            child.shapes.append(parent1.shapes[i].copy())
        else:
            child.shapes.append(parent2.shapes[i].copy())
    return child


def crossover_concat(parent1: Genotype, parent2: Genotype) -> Genotype:
    """
    Concatenation: Child inherits ALL shapes from BOTH parents.
    """
    child = Genotype(parent1.w, parent1.h)
    child.shapes = [s.copy() for s in parent1.shapes] + [s.copy() for s in parent2.shapes]
    return child


def apply_crossover(parent1: Genotype, parent2: Genotype, method: str = "uniform") -> Genotype:
    """Dispatcher for crossover methods."""
    if method == "concat":
        return crossover_concat(parent1, parent2)
    else:
        return crossover_uniform(parent1, parent2)


# =============================================================================
# Genetic Algorithms
# =============================================================================

def genetic_algorithm(
    original_img: np.ndarray,
    shape_class: type[SVGShape],
    pop_size: int = 50,
    max_it: int = 1000,
    selection_method: Literal["greedy", "tournament"] = "tournament",
    crossover_type: str = "uniform",
    mutation_rate: float = 0.5,
    track_by_time: bool = False,
    use_opacity: bool = False,
    fitness_fn=None,
    use_heuristic: bool = False,
    max_time: float = None,
) -> tuple[Genotype, list[tuple[float, float]]]:
    """
    Genetic Algorithm with configurable selection method.
    
    Args:
        original_img: Target image as numpy array
        shape_class: Shape class to use (Rect, Circle, Ellipse)
        pop_size: Population size
        max_it: Maximum generations
        selection_method: 'greedy' (truncation) or 'tournament'
        crossover_type: 'uniform' or 'concat'
        mutation_rate: Probability of mutation per child
        track_by_time: If True, track fitness by elapsed seconds; otherwise by iteration
        use_opacity: If True, enable random opacity mutations
        fitness_fn: Fitness function to use (default: compute_fitness L2)
        use_heuristic: If True, initialize population with grid-based heuristic
        max_time: Maximum time in seconds (None = no time limit)
        
    Returns:
        Tuple of (best_genotype, history) where history is list of (iteration_or_time, fitness)
    """
    if fitness_fn is None:
        fitness_fn = compute_fitness
        
    w, h = original_img.shape[1], original_img.shape[0]
    
    # Initialize population
    population: list[Genotype] = []
    
    if use_heuristic:
        print("Initializing Population with heuristic grid...")
        for _ in range(pop_size):
            # Create jittered grid individuals with varied grid sizes
            grid_size = random.choice([3, 4, 5])
            geno = get_jittered_grid_individual(
                w, h, shape_class, grid_size=grid_size,
                jitter_amount=10, original_img=original_img,
                use_opacity=use_opacity
            )
            population.append(geno)
    else:
        print("Initializing Population...")
        for _ in range(pop_size):
            geno = Genotype(w, h)
            geno.shapes.append(get_random_shape(shape_class, w, h, use_opacity=use_opacity))
            population.append(geno)
    
    best_global_score = float('inf')
    best_global_geno = None
    history: list[tuple[float, float]] = []
    start_time = time.time()
    
    pbar = tqdm(range(max_it), desc=f"GA ({selection_method})")
    
    for iteration in pbar:
        # Check time limit
        elapsed = time.time() - start_time
        if max_time is not None and elapsed >= max_time:
            print(f"Stopping at generation {iteration} - time limit reached ({max_time:.1f}s)")
            break
            
        # Evaluation Phase - only compute fitness if missing
        for geno in population:
            if geno.fitness is None:
                pheno = generate_phenotype(w, h, geno.shapes)
                geno.fitness = fitness_fn(original_img, pheno)
        
        # Sort by fitness (lowest is best)
        population.sort(key=lambda g: g.fitness)
        
        # Elitism - track best
        best_current = population[0]
        
        if best_current.fitness < best_global_score:
            best_global_score = best_current.fitness
            best_global_geno = best_current.clone()
        
        # Record history
        if track_by_time:
            history.append((time.time() - start_time, best_global_score))
        else:
            history.append((float(iteration), best_global_score))
        
        pbar.set_postfix({"Best": f"{best_global_score:,}", "Shapes": len(best_current.shapes)})
        
        # Create next generation with elitism
        next_gen: list[Genotype] = [best_current.clone()]
        
        while len(next_gen) < pop_size:
            # Selection
            if selection_method == "tournament":
                k = 3
                p1 = min(random.sample(population, k), key=lambda g: g.fitness)
                p2 = min(random.sample(population, k), key=lambda g: g.fitness)
                parent1, parent2 = p1, p2
            else:  # greedy
                top_k = max(2, int(pop_size * 0.2))
                top_pool = population[:top_k]
                parent1 = random.choice(top_pool)
                parent2 = random.choice(top_pool)
            
            # Crossover
            child = apply_crossover(parent1, parent2, method=crossover_type)
            
            # Mutation
            if random.random() < mutation_rate:
                apply_mutation(
                    child,
                    shape_class,
                    original_img=original_img,
                    mutation_type="random",
                    delta_scale=0.1,
                    move_range=15,
                    use_opacity=use_opacity,
                )
                child.fitness = None  # Invalidate fitness after mutation
            
            next_gen.append(child)
        
        population = next_gen
    
    return best_global_geno, history


def ga_greedy(
    original_img: np.ndarray,
    shape_class: type[SVGShape],
    pop_size: int = 50,
    max_it: int = 1000,
    mutation_rate: float = 0.5,
    track_by_time: bool = False,
    use_opacity: bool = False,
    fitness_fn=None,
    use_heuristic: bool = False,
    max_time: float = None,
) -> tuple[Genotype, list[tuple[float, float]]]:
    """Genetic Algorithm with Greedy (Truncation) selection."""
    return genetic_algorithm(
        original_img=original_img,
        shape_class=shape_class,
        pop_size=pop_size,
        max_it=max_it,
        selection_method="greedy",
        mutation_rate=mutation_rate,
        track_by_time=track_by_time,
        use_opacity=use_opacity,
        fitness_fn=fitness_fn,
        use_heuristic=use_heuristic,
        max_time=max_time,
    )


def ga_tournament(
    original_img: np.ndarray,
    shape_class: type[SVGShape],
    pop_size: int = 50,
    max_it: int = 1000,
    mutation_rate: float = 0.5,
    track_by_time: bool = False,
    use_opacity: bool = False,
    fitness_fn=None,
    use_heuristic: bool = False,
    max_time: float = None,
) -> tuple[Genotype, list[tuple[float, float]]]:
    """Genetic Algorithm with Tournament selection."""
    return genetic_algorithm(
        original_img=original_img,
        shape_class=shape_class,
        pop_size=pop_size,
        max_it=max_it,
        selection_method="tournament",
        mutation_rate=mutation_rate,
        track_by_time=track_by_time,
        use_opacity=use_opacity,
        fitness_fn=fitness_fn,
        use_heuristic=use_heuristic,
        max_time=max_time,
    )
