from simpleai.search import SearchProblem, astar, greedy
import time

class EightPuzzleProblem(SearchProblem):    
    def __init__(self, initial_state=None, goal_state=None):
        if goal_state is None:
            goal_state = ((1, 2, 3),
                         (8, 0, 4),
                         (7, 6, 5))
        
        if initial_state is None:
            initial_state = ((2, 8, 3),
                            (1, 6, 4),
                            (7, 0, 5))

        self.goal_state = goal_state
        super().__init__(initial_state)
    
    def _get_valid_actions(self, state):
        """Get valid actions for current state"""
        actions = []
        empty_row, empty_col = self._find_empty(state)
        
        if empty_row > 0:
            actions.append('UP')
        if empty_row < 2:
            actions.append('DOWN')
        if empty_col > 0:
            actions.append('LEFT')
        if empty_col < 2:
            actions.append('RIGHT')
        
        return actions
    
    def _apply_action(self, state, action):
        """Apply action to state and return new state"""
        new_state = [list(row) for row in state]
        empty_row, empty_col = self._find_empty(state)
        
        if action == 'UP':
            new_state[empty_row][empty_col] = new_state[empty_row-1][empty_col]
            new_state[empty_row-1][empty_col] = 0
        elif action == 'DOWN':
            new_state[empty_row][empty_col] = new_state[empty_row+1][empty_col]
            new_state[empty_row+1][empty_col] = 0
        elif action == 'LEFT':
            new_state[empty_row][empty_col] = new_state[empty_row][empty_col-1]
            new_state[empty_row][empty_col-1] = 0
        elif action == 'RIGHT':
            new_state[empty_row][empty_col] = new_state[empty_row][empty_col+1]
            new_state[empty_row][empty_col+1] = 0
        
        return tuple(tuple(row) for row in new_state)
    
    def actions(self, state):
        actions = []
        empty_row, empty_col = self._find_empty(state)
        
        if empty_row > 0:
            actions.append('UP')
        
        if empty_row < 2:
            actions.append('DOWN')
        
        if empty_col > 0:
            actions.append('LEFT')
        
        if empty_col < 2:
            actions.append('RIGHT')
        
        return actions
    
    def result(self, state, action):
        # Chuyển tuple thành list để có thể sửa đổi
        new_state = [list(row) for row in state]
        empty_row, empty_col = self._find_empty(state)
        
        if action == 'UP':
            new_state[empty_row][empty_col] = new_state[empty_row-1][empty_col]
            new_state[empty_row-1][empty_col] = 0
        elif action == 'DOWN':
            new_state[empty_row][empty_col] = new_state[empty_row+1][empty_col]
            new_state[empty_row+1][empty_col] = 0
        elif action == 'LEFT':
            new_state[empty_row][empty_col] = new_state[empty_row][empty_col-1]
            new_state[empty_row][empty_col-1] = 0
        elif action == 'RIGHT':
            new_state[empty_row][empty_col] = new_state[empty_row][empty_col+1]
            new_state[empty_row][empty_col+1] = 0
        
        # Chuyển về tuple
        return tuple(tuple(row) for row in new_state)
    
    def is_goal(self, state):
        return state == self.goal_state
    
    def cost(self, state1, action, state2):
        return 1
    
    def heuristic(self, state):
        """Simple Manhattan distance for reliable performance"""
        return self._manhattan_distance(state)
    
    def _find_empty(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j
        return None
    
    def _manhattan_distance(self, state):
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    value = state[i][j]
                    goal_row, goal_col = self._find_goal_position(value)
                    distance += abs(i - goal_row) + abs(j - goal_col)
        return distance
    
    def _find_goal_position(self, value):
        for i in range(3):
            for j in range(3):
                if self.goal_state[i][j] == value:
                    return i, j
        return None

def print_board(state, title="State"):
    print(f"\n{title}:")
    print("┌─────────┐")
    for row in state:
        print("│", end="")
        for cell in row:
            if cell == 0:
                print("   ", end="")  
            else:
                print(f" {cell} ", end="")
        print("│")
    print("└─────────┘")

def print_solution_path(result, title="Solution Path"):
    if result is None:
        print("No solution found!")
        return
    
    path = result.path()
    print(f"\n{title}:")
    print(f"Solution found with {len(path)} steps")
    
    show_path = input("Show complete solution path? (y/n, default=n): ").strip().lower() == 'y'
    
    if show_path:
        for i, (action, state) in enumerate(path):
            if i == 0:
                print_board(state, f"Initial State")
            else:
                print(f"\nAction: {action}")
                print_board(state, f"Step {i}")
    else:
        print_board(path[0][1], "Initial State")
        print("...")
        print_board(path[-1][1], "Final State")

def solve_with_astar():
    """Solve with A* algorithm"""
    print("\n### Solving 8-Puzzle with A* Search ###")
    problem = EightPuzzleProblem()
    
    print("Initial state:")
    print_board(problem.initial_state, "Initial")
    
    print("Goal state:")
    print_board(problem.goal_state, "Goal")
    
    print(f"Manhattan distance: {problem.heuristic(problem.initial_state)}")
    
    print("\nSolving with A*...")
    start_time = time.time()
    result = astar(problem)
    end_time = time.time()
    
    if result is not None:
        print(f"A* found solution!")
        print(f"Time: {end_time - start_time:.4f} seconds")
        print(f"Path length: {len(result.path())} steps")
        print(f"Total cost: {result.cost}")
        print_solution_path(result, "A* Solution")
    else:
        print("A* could not find solution!")
    
    return result, end_time - start_time

def solve_with_greedy():
    """Solve with Greedy algorithm"""
    print("\n### Solving 8-Puzzle with Greedy Search ###")
    problem = EightPuzzleProblem()
    
    print("Initial state:")
    print_board(problem.initial_state, "Initial")
    
    print("Goal state:")
    print_board(problem.goal_state, "Goal")
    
    print(f"Manhattan distance: {problem.heuristic(problem.initial_state)}")
    
    print("\nSolving with Greedy...")
    start_time = time.time()
    result = greedy(problem)
    end_time = time.time()
    
    if result is not None:
        print(f"Greedy found solution!")
        print(f"Time: {end_time - start_time:.4f} seconds")
        print(f"Path length: {len(result.path())} steps")
        print(f"Total cost: {result.cost}")
        print_solution_path(result, "Greedy Solution")
    else:
        print("Greedy could not find solution!")

    return result, end_time - start_time

def display_menu():
    print("\n" + "=" * 50)
    print("8-PUZZLE SOLVER")
    print("=" * 50)
    print("\n1. Solve with A* algorithm")
    print("2. Solve with Greedy algorithm")
    print("0. Exit")
    print("-" * 50)

def main():
    while True:
        display_menu()
        choice = input("\nEnter your choice (0-2): ").strip()
        
        if choice == '1':
            solve_with_astar()
        elif choice == '2':
            solve_with_greedy()
        elif choice == '0':
            print("\nThank you for using 8-Puzzle Solver!")
            break
        else:
            print("\nInvalid choice! Please enter a number between 0-4.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()