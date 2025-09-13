import random
import time
from simpleai.search import SearchProblem, hill_climbing, genetic, simulated_annealing

class EightQueensProblem(SearchProblem):
    
    def __init__(self, initial_state=None):
        if initial_state is None:
            initial_state = tuple(random.randint(0, 7) for _ in range(8))
        super().__init__(initial_state)
    
    def actions(self, state):
        actions = []
        for row in range(8):
            for col in range(8):
                if state[row] != col:  
                    actions.append((row, col))
        return actions
    
    def result(self, state, action):
        row, new_col = action
        new_state = list(state)
        new_state[row] = new_col
        return tuple(new_state)
    
    def value(self, state):
        return 28 - self._conflicts(state)
    
    def _conflicts(self, state):
        conflicts = 0
        for i in range(8):
            for j in range(i + 1, 8):
                if state[i] == state[j]:
                    conflicts += 1
                elif abs(state[i] - state[j]) == abs(i - j):
                    conflicts += 1
        return conflicts
    
    def is_goal(self, state):
        return self._conflicts(state) == 0

def generate_random_state():
    return tuple(random.randint(0, 7) for _ in range(8))

def crossover(state1, state2):
    crossover_point = random.randint(1, 6)
    child1 = state1[:crossover_point] + state2[crossover_point:]
    child2 = state2[:crossover_point] + state1[crossover_point:]
    return child1, child2

def mutate(state):
    state_list = list(state)
    row = random.randint(0, 7)
    new_col = random.randint(0, 7)
    state_list[row] = new_col
    return tuple(state_list)

def print_board(state):
    print("Bàn cờ 8x8 với các quân hậu:")
    print("  " + " ".join(str(i) for i in range(8)))
    for row in range(8):
        line = f"{row} "
        for col in range(8):
            if state[row] == col:
                line += "Q "
            else:
                line += ". "
        print(line)
    print()

def solve_with_hill_climbing():
    print("=== HILL CLIMBING ===")
    problem = EightQueensProblem()
    print(f"State ban đầu: {problem.initial_state}")
    print(f"Số conflicts ban đầu: {problem._conflicts(problem.initial_state)}")
    print_board(problem.initial_state)
    
    start_time = time.time()
    result = hill_climbing(problem)
    end_time = time.time()
    
    if result:
        print(f"Tìm thấy nghiệm: {result.state}")
        print(f"Số conflicts: {problem._conflicts(result.state)}")
        print_board(result.state)
    else:
        print("Không tìm thấy nghiệm (có thể bị kẹt ở local maximum)")
    
    print(f"Thời gian thực hiện: {end_time - start_time:.4f} giây")
    print("-" * 50)
    return result

def solve_with_genetic():
    print("=== GENETIC ALGORITHM ===")
    problem = EightQueensProblem()
    
    population_size = 100
    mutation_rate = 0.1
    generations = 1000
    
    start_time = time.time()
    population = [generate_random_state() for _ in range(population_size)]
    
    best_solution = None
    best_fitness = -1
    
    for generation in range(generations):
        fitness_scores = [problem.value(individual) for individual in population]
        
        max_fitness = max(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_index = fitness_scores.index(max_fitness)
            best_solution = population[best_index]
            
            if max_fitness == 28:
                break
        
        new_population = []
        
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            
            child1, child2 = crossover(parent1, parent2)
            
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            
            new_population.extend([child1, child2])
        
        population = new_population
    
    end_time = time.time()
    
    if best_solution and problem._conflicts(best_solution) == 0:
        print(f"Tìm thấy nghiệm: {best_solution}")
        print(f"Số conflicts: {problem._conflicts(best_solution)}")
        print_board(best_solution)
    else:
        print("Không tìm thấy nghiệm hoàn hảo")
        if best_solution:
            print(f"Nghiệm tốt nhất: {best_solution}")
            print(f"Số conflicts: {problem._conflicts(best_solution)}")
            print_board(best_solution)
    
    print(f"Thời gian thực hiện: {end_time - start_time:.4f} giây")
    print("-" * 50)
    
    class Result:
        def __init__(self, state):
            self.state = state
    
    return Result(best_solution) if best_solution else None

def tournament_selection(population, fitness_scores, tournament_size=3):
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_index]

def solve_with_simulated_annealing():
    print("=== SIMULATED ANNEALING ===")
    problem = EightQueensProblem()
    print(f"State ban đầu: {problem.initial_state}")
    print(f"Số conflicts ban đầu: {problem._conflicts(problem.initial_state)}")
    print_board(problem.initial_state)
    
    start_time = time.time()
    result = simulated_annealing(problem, iterations_limit=10000)
    end_time = time.time()
    
    if result:
        print(f"Tìm thấy nghiệm: {result.state}")
        print(f"Số conflicts: {problem._conflicts(result.state)}")
        print_board(result.state)
    else:
        print("Không tìm thấy nghiệm")
    
    print(f"Thời gian thực hiện: {end_time - start_time:.4f} giây")
    print("-" * 50)
    return result

def compare_algorithms():
    print("=== SO SÁNH CÁC THUẬT TOÁN ===")
    
    algorithms = [
        ("Hill Climbing", solve_with_hill_climbing),
        ("Genetic Algorithm", solve_with_genetic),
        ("Simulated Annealing", solve_with_simulated_annealing)
    ]
    
    results = {}
    
    for name, algorithm in algorithms:
        print(f"\nChạy {name}...")
        successes = 0
        total_time = 0
        runs = 5 
        
        for i in range(runs):
            start = time.time()
            result = algorithm()
            end = time.time()
            
            if result and EightQueensProblem()._conflicts(result.state) == 0:
                successes += 1
            total_time += (end - start)
        
        success_rate = (successes / runs) * 100
        avg_time = total_time / runs
        
        results[name] = {
            'success_rate': success_rate,
            'avg_time': avg_time
        }
        
        print(f"{name}: Tỷ lệ thành công: {success_rate}%, Thời gian trung bình: {avg_time:.4f}s")
    
    return results

def solve_with_fixed_initial_state():
    print("GIẢI BÀI TOÁN 8 QUÂN HẬU BẰNG SIMPLEAI")
    print("=" * 60)
    
    initial_state = (0, 1, 2, 3, 4, 5, 6, 7)
    print(f"State đầu cố định cho cả 3 thuật toán: {initial_state}")
    problem_temp = EightQueensProblem(initial_state)
    print(f"Số conflicts ban đầu: {problem_temp._conflicts(initial_state)}")
    print_board(initial_state)
    
    results = {}
    
    print("=== 1. HILL CLIMBING ===")
    problem1 = EightQueensProblem(initial_state)
    start_time = time.time()
    result1 = hill_climbing(problem1)
    end_time = time.time()
    
    if result1:
        print(f"Tìm thấy nghiệm: {result1.state}")
        print(f"Số conflicts: {problem1._conflicts(result1.state)}")
        print_board(result1.state)
        results['Hill Climbing'] = {
            'success': problem1._conflicts(result1.state) == 0,
            'final_state': result1.state,
            'conflicts': problem1._conflicts(result1.state),
            'time': end_time - start_time
        }
    else:
        print("Không tìm thấy nghiệm (có thể bị kẹt ở local maximum)")
        results['Hill Climbing'] = {
            'success': False,
            'final_state': None,
            'conflicts': float('inf'),
            'time': end_time - start_time
        }
    
    print(f"Thời gian thực hiện: {end_time - start_time:.4f} giây")
    print("-" * 50)
    
    print("=== 2. GENETIC ALGORITHM ===")
    problem2 = EightQueensProblem(initial_state)
    
    population_size = 100
    mutation_rate = 0.1
    generations = 1000
    
    start_time = time.time()
    population = [initial_state] + [generate_random_state() for _ in range(population_size - 1)]
    
    best_solution = None
    best_fitness = -1
    
    for generation in range(generations):
        fitness_scores = [problem2.value(individual) for individual in population]
        
        max_fitness = max(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_index = fitness_scores.index(max_fitness)
            best_solution = population[best_index]
            
            if max_fitness == 28:
                break
        
        new_population = []
        
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            
            child1, child2 = crossover(parent1, parent2)
            
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            
            new_population.extend([child1, child2])
        
        population = new_population
    
    end_time = time.time()
    
    if best_solution and problem2._conflicts(best_solution) == 0:
        print(f"Tìm thấy nghiệm: {best_solution}")
        print(f"Số conflicts: {problem2._conflicts(best_solution)}")
        print_board(best_solution)
        results['Genetic Algorithm'] = {
            'success': True,
            'final_state': best_solution,
            'conflicts': problem2._conflicts(best_solution),
            'time': end_time - start_time
        }
    else:
        print("Không tìm thấy nghiệm hoàn hảo")
        if best_solution:
            print(f"Nghiệm tốt nhất: {best_solution}")
            print(f"Số conflicts: {problem2._conflicts(best_solution)}")
            print_board(best_solution)
            results['Genetic Algorithm'] = {
                'success': False,
                'final_state': best_solution,
                'conflicts': problem2._conflicts(best_solution),
                'time': end_time - start_time
            }
        else:
            results['Genetic Algorithm'] = {
                'success': False,
                'final_state': None,
                'conflicts': float('inf'),
                'time': end_time - start_time
            }
    
    print(f"Thời gian thực hiện: {end_time - start_time:.4f} giây")
    print("-" * 50)

    print("=== 3. SIMULATED ANNEALING ===")
    problem3 = EightQueensProblem(initial_state)
    start_time = time.time()
    result3 = simulated_annealing(problem3, iterations_limit=10000)
    end_time = time.time()
    
    if result3:
        print(f"Tìm thấy nghiệm: {result3.state}")
        print(f"Số conflicts: {problem3._conflicts(result3.state)}")
        print_board(result3.state)
        results['Simulated Annealing'] = {
            'success': problem3._conflicts(result3.state) == 0,
            'final_state': result3.state,
            'conflicts': problem3._conflicts(result3.state),
            'time': end_time - start_time
        }
    else:
        print("Không tìm thấy nghiệm")
        results['Simulated Annealing'] = {
            'success': False,
            'final_state': None,
            'conflicts': float('inf'),
            'time': end_time - start_time
        }
    
    print(f"Thời gian thực hiện: {end_time - start_time:.4f} giây")
    print("-" * 50)
    
    print("=== SO SÁNH KẾT QUẢ ===")
    print(f"{'Thuật toán':<20} {'Thành công':<12} {'Conflicts':<10} {'Thời gian (s)':<12}")
    print("-" * 55)
    
    for name, result in results.items():
        success_str = "✓" if result['success'] else "✗"
        conflicts = result['conflicts'] if result['conflicts'] != float('inf') else "N/A"
        print(f"{name:<20} {success_str:<12} {conflicts:<10} {result['time']:<12.4f}")

if __name__ == "__main__":
    solve_with_fixed_initial_state()

