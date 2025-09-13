from simpleai.search import SearchProblem, astar, greedy, breadth_first, depth_first
import time

class EightQueensProblem(SearchProblem):
    def __init__(self, initial_state=None):
        if initial_state is None:
            initial_state = tuple()
        super().__init__(initial_state)
        self.size = 8
    
    def actions(self, state):
        if len(state) == self.size:
            return []
        
        valid_positions = []
        row = len(state)
        
        for col in range(self.size):
            if self._is_safe(state, row, col):
                valid_positions.append(col)
        
        return valid_positions
    
    def result(self, state, action):
        return state + (action,)
    
    def is_goal(self, state):
        return len(state) == self.size
    
    def _is_safe(self, state, row, col):
        for prev_row, prev_col in enumerate(state):
            if prev_col == col:
                return False
            
            if abs(prev_row - row) == abs(prev_col - col):
                return False
        
        return True
    
    def heuristic(self, state):
        if self.is_goal(state):
            return 0
        
        remaining_queens = self.size - len(state)
        
        conflicts = 0
        for i, col1 in enumerate(state):
            for j, col2 in enumerate(state[i+1:], i+1):
                if abs(i - j) == abs(col1 - col2):
                    conflicts += 1
        
        next_row = len(state)
        available_cols = 0
        for col in range(self.size):
            if self._is_safe(state, next_row, col):
                available_cols += 1
        
        if available_cols == 0 and remaining_queens > 0:
            return float('inf')
        
        return remaining_queens + conflicts + (self.size - available_cols)
    
    def cost(self, state1, action, state2):
        return 1

def print_board(solution, title="Solution"):
    if solution is None:
        print("No solution found!")
        return
    
    if hasattr(solution, 'state'):
        state = solution.state
    else:
        state = solution
    
    size = len(state)
    
    print(f"\n{title}:")
    print("=" * (size * 4 + 1))
    for row in range(size):
        line = "|"
        for col in range(size):
            if state[row] == col:
                line += " Q |"
            else:
                line += "   |"
        print(line)
        print("-" * (size * 4 + 1))
    print("=" * (size * 4 + 1))
    print(f"Queen positions (row, col): {[(i, state[i]) for i in range(len(state))]}")

def find_all_solutions():
    print("\n### Finding ALL 8 Queens Solutions ###")
    print("This will find all 92 possible solutions using DFS...")
    
    all_solutions = []
    visited = set()
    
    def dfs_all(state):
        if len(state) == 8:
            if state not in visited:
                visited.add(state)
                all_solutions.append(state)
            return
        
        row = len(state)
        for col in range(8):
            if is_safe_position(state, row, col):
                dfs_all(state + (col,))
    
    def is_safe_position(state, row, col):
        for prev_row, prev_col in enumerate(state):
            if prev_col == col or abs(prev_row - row) == abs(prev_col - col):
                return False
        return True
    
    start_time = time.time()
    dfs_all(tuple())
    end_time = time.time()
    
    print(f"\nFound {len(all_solutions)} unique solutions in {end_time - start_time:.4f} seconds")
    
    show_all = input("\nShow all solutions? (y/n, default=n): ").strip().lower() == 'y'
    
    if show_all:
        for i, solution in enumerate(all_solutions, 1):
            print_board(solution, f"Solution #{i}")
    else:
        if all_solutions:
            print("\nShowing first 3 solutions as examples:")
            for i in range(min(3, len(all_solutions))):
                print_board(all_solutions[i], f"Solution #{i+1}")
    
    return all_solutions

def solve_first_with_astar():
    print("\n### Finding FIRST Solution with A* Search ###")
    problem = EightQueensProblem()
    
    start_time = time.time()
    result = astar(problem)
    end_time = time.time()
    
    if result is not None:
        print(f"\nSolution found in {end_time - start_time:.4f} seconds")
        print(f"Path length: {len(result.path())}")
        print(f"Solution state: {result.state}")
        print_board(result, "A* Solution")
    else:
        print("No solution found!")
    
    return result, end_time - start_time

def solve_first_with_greedy():
    print("\n### Finding FIRST Solution with Greedy Search ###")
    problem = EightQueensProblem()
    
    start_time = time.time()
    result = greedy(problem)
    end_time = time.time()
    
    if result is not None:
        print(f"\nSolution found in {end_time - start_time:.4f} seconds")
        print(f"Path length: {len(result.path())}")
        print(f"Solution state: {result.state}")
        print_board(result, "Greedy Solution")
    else:
        print("No solution found!")
    
    return result, end_time - start_time

def compare_first_solutions():
    print("\n" + "=" * 60)
    print("COMPARING A* AND GREEDY SEARCH FOR FIRST SOLUTION")
    print("=" * 60)
    
    astar_result, astar_time = solve_first_with_astar()
    greedy_result, greedy_time = solve_first_with_greedy()
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\nA* Search:")
    print(f"  - Time: {astar_time:.4f} seconds")
    if astar_result:
        print(f"  - Solution found: Yes")
        print(f"  - Path length: {len(astar_result.path())}")
        print(f"  - Solution: {astar_result.state}")
    else:
        print(f"  - Solution found: No")
    
    print(f"\nGreedy Search:")
    print(f"  - Time: {greedy_time:.4f} seconds")
    if greedy_result:
        print(f"  - Solution found: Yes")
        print(f"  - Path length: {len(greedy_result.path())}")
        print(f"  - Solution: {greedy_result.state}")
    else:
        print(f"  - Solution found: No")
    
    if astar_time < greedy_time:
        print(f"\nA* was faster by {greedy_time - astar_time:.4f} seconds")
    else:
        print(f"\nGreedy was faster by {astar_time - greedy_time:.4f} seconds")

def find_all_with_algorithm(algorithm_name):
    print(f"\n### Finding ALL Solutions with Modified {algorithm_name} ###")
    print("Note: Traditional A*/Greedy return first solution only.")
    print("Using systematic search to find all solutions...\n")
    
    all_solutions = []
    visited = set()
    
    if algorithm_name == "A*":
        print("Using A* heuristic to guide systematic search...")
        problem = EightQueensProblem()
        
        def guided_dfs(state, heuristic_func):
            if len(state) == 8:
                if state not in visited:
                    visited.add(state)
                    all_solutions.append(state)
                return
            
            row = len(state)
            actions = []
            for col in range(8):
                if is_safe_position(state, row, col):
                    next_state = state + (col,)
                    h_value = heuristic_func(next_state)
                    actions.append((col, h_value))
            
            actions.sort(key=lambda x: x[1])
            
            for col, _ in actions:
                guided_dfs(state + (col,), heuristic_func)
        
        def is_safe_position(state, row, col):
            for prev_row, prev_col in enumerate(state):
                if prev_col == col or abs(prev_row - row) == abs(prev_col - col):
                    return False
            return True
        
        start_time = time.time()
        guided_dfs(tuple(), problem.heuristic)
        end_time = time.time()
        
    else:  
        print("Using Greedy approach to explore all branches...")
        
        def greedy_dfs(state):
            if len(state) == 8:
                if state not in visited:
                    visited.add(state)
                    all_solutions.append(state)
                return
            
            row = len(state)
            for col in range(8):
                if is_safe_position(state, row, col):
                    greedy_dfs(state + (col,))
        
        def is_safe_position(state, row, col):
            for prev_row, prev_col in enumerate(state):
                if prev_col == col or abs(prev_row - row) == abs(prev_col - col):
                    return False
            return True
        
        start_time = time.time()
        greedy_dfs(tuple())
        end_time = time.time()
    
    print(f"Found {len(all_solutions)} unique solutions in {end_time - start_time:.4f} seconds")
    
    show_all = input("\nShow all solutions? (y/n, default=n): ").strip().lower() == 'y'
    
    if show_all:
        for i, solution in enumerate(all_solutions, 1):
            print_board(solution, f"Solution #{i}")
    else:
        if all_solutions:
            print("\nShowing first 3 solutions as examples:")
            for i in range(min(3, len(all_solutions))):
                print_board(all_solutions[i], f"Solution #{i+1}")
    
    return all_solutions

def display_menu():
    print("\n" + "=" * 60)
    print("8 QUEENS PROBLEM SOLVER")
    print("=" * 60)
    print("\n1. Find FIRST solution with A*")
    print("2. Find FIRST solution with Greedy")
    print("3. Compare A* and Greedy (FIRST solution)")
    print("4. Find ALL 92 solutions (standard DFS)")
    print("5. Find ALL solutions with A* guided search")
    print("6. Find ALL solutions with Greedy approach")
    print("0. Exit")
    print("-" * 60)

def main():
    while True:
        display_menu()
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '1':
            solve_first_with_astar()
        elif choice == '2':
            solve_first_with_greedy()
        elif choice == '3':
            compare_first_solutions()
        elif choice == '4':
            find_all_solutions()
        elif choice == '5':
            find_all_with_algorithm("A*")
        elif choice == '6':
            find_all_with_algorithm("Greedy")
        elif choice == '0':
            print("\nThank you for using 8 Queens Solver!")
            break
        else:
            print("\nInvalid choice! Please enter a number between 0-6.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()