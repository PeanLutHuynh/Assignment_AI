# Bài 3/ Bài toán 8-puzzle (ô chữ 8 số)

from collections import deque
import heapq
import copy

# Goal state: ma trận 3x3 với ô trống ở giữa
GOAL_STATE = [
    [1, 2, 3],
    [8, 0, 4],
    [7, 6, 5]
]

class PuzzleProblem:
    """Bài toán 8-puzzle với ma trận 3x3"""
    
    def __init__(self, initial_state=None):
        if initial_state is None:
            # Initial state: chỉ thay đổi vài số từ goal để dễ giải
            self.initial_state = [
                [1, 4, 0],
                [5, 2, 3],  # Đổi chỗ 0 và 4
                [7, 6, 8]
            ]
        else:
            self.initial_state = initial_state
    
    def actions(self, state):
        """Trả về các hành động có thể: di chuyển ô trống lên/xuống/trái/phải"""
        actions = []
        empty_row, empty_col = self.find_empty(state)
        
        # Có thể di chuyển lên
        if empty_row > 0:
            actions.append('UP')
        # Có thể di chuyển xuống  
        if empty_row < 2:
            actions.append('DOWN')
        # Có thể di chuyển trái
        if empty_col > 0:
            actions.append('LEFT')
        # Có thể di chuyển phải
        if empty_col < 2:
            actions.append('RIGHT')
            
        return actions
    
    def result(self, state, action):
        """Trả về trạng thái mới sau khi thực hiện action"""
        new_state = copy.deepcopy(state)
        empty_row, empty_col = self.find_empty(state)
        
        if action == 'UP':
            # Đổi chỗ ô trống với ô phía trên
            new_state[empty_row][empty_col] = new_state[empty_row-1][empty_col]
            new_state[empty_row-1][empty_col] = 0
        elif action == 'DOWN':
            # Đổi chỗ ô trống với ô phía dưới
            new_state[empty_row][empty_col] = new_state[empty_row+1][empty_col]
            new_state[empty_row+1][empty_col] = 0
        elif action == 'LEFT':
            # Đổi chỗ ô trống với ô bên trái
            new_state[empty_row][empty_col] = new_state[empty_row][empty_col-1]
            new_state[empty_row][empty_col-1] = 0
        elif action == 'RIGHT':
            # Đổi chỗ ô trống với ô bên phải
            new_state[empty_row][empty_col] = new_state[empty_row][empty_col+1]
            new_state[empty_row][empty_col+1] = 0
            
        return new_state
    
    def is_goal(self, state):
        """Kiểm tra xem có phải goal state không"""
        return state == GOAL_STATE
    
    def get_cost(self, state, action):
        """Chi phí mỗi action = 1"""
        return 1
    
    def find_empty(self, state):
        """Tìm vị trí ô trống (số 0)"""
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j
        return None
    
    def state_to_string(self, state):
        """Chuyển state thành string để dễ so sánh và hash"""
        return str(state)
    
    def print_state(self, state):
        """In ma trận một cách đẹp mắt"""
        print("┌─────────┐")
        for row in state:
            print("│", end="")
            for cell in row:
                if cell == 0:
                    print("   ", end="")  # Ô trống
                else:
                    print(f" {cell} ", end="")
            print("│")
        print("└─────────┘")

# Heuristic cho 8-puzzle: Manhattan distance
def manhattan_distance(state):
    """Tính Manhattan distance từ state hiện tại đến goal"""
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:  # Bỏ qua ô trống
                value = state[i][j]
                # Tìm vị trí đích của value trong goal
                goal_row, goal_col = find_goal_position(value)
                distance += abs(i - goal_row) + abs(j - goal_col)
    return distance

def find_goal_position(value):
    """Tìm vị trí của value trong goal state"""
    for i in range(3):
        for j in range(3):
            if GOAL_STATE[i][j] == value:
                return i, j
    return None

# Các thuật toán TỐI ƯU với HEURISTIC cho 8-puzzle

#1. BFS với heuristic
def bfs_puzzle():
    """BFS cho 8-puzzle"""
    problem = PuzzleProblem()
    start = problem.initial_state
    
    if problem.is_goal(start):
        return [start]
    
    queue = deque([(start, [start])])
    visited = set([problem.state_to_string(start)])
    nodes_explored = 0
    
    while queue:
        current, path = queue.popleft()
        nodes_explored += 1
        
        # Early stopping
        if nodes_explored > 1000000:
            print(f"BFS dừng sớm sau {nodes_explored} nodes")
            break
            
        for action in problem.actions(current):
            next_state = problem.result(current, action)
            state_str = problem.state_to_string(next_state)
            
            if state_str not in visited:
                new_path = path + [next_state]
                
                if problem.is_goal(next_state):
                    return new_path
                
                queue.append((next_state, new_path))
                visited.add(state_str)
    return None

# 2. UCS với heuristic
def ucs_puzzle():
    """UCS cho 8-puzzle"""
    problem = PuzzleProblem()
    start = problem.initial_state
    
    if problem.is_goal(start):
        return [start], 0
    
    heap = [(0, problem.state_to_string(start), start, [start])]
    visited = set()
    nodes_explored = 0
    
    while heap:
        cost, state_str, current, path = heapq.heappop(heap)
        nodes_explored += 1
        
        # Early stopping
        if nodes_explored > 1000000:
            print(f"UCS dừng sớm sau {nodes_explored} nodes")
            break
            
        if state_str in visited:
            continue
        visited.add(state_str)
        
        if problem.is_goal(current):
            return path, cost
        
        for action in problem.actions(current):
            next_state = problem.result(current, action)
            next_state_str = problem.state_to_string(next_state)
            
            if next_state_str not in visited:
                new_cost = cost + problem.get_cost(current, action)
                new_path = path + [next_state]
                heapq.heappush(heap, (new_cost, next_state_str, next_state, new_path))
    
    return None, float('inf')

# 3. DFS với heuristic
def dfs_puzzle(current=None, path=None, visited=None, depth=0, max_depth=20):
    """DFS cho 8-puzzle với giới hạn độ sâu"""
    problem = PuzzleProblem()
    
    if current is None:
        current = problem.initial_state
    if path is None:
        path = [current]
    if visited is None:
        visited = set()
    
    if problem.is_goal(current):
        return path
    
    state_str = problem.state_to_string(current)
    if depth >= max_depth or state_str in visited:
        return None
        
    visited.add(state_str)
    
    for action in problem.actions(current):
        next_state = problem.result(current, action)
        next_state_str = problem.state_to_string(next_state)
        
        if next_state_str not in visited:
            result = dfs_puzzle(next_state, path + [next_state], visited.copy(), depth + 1, max_depth)
            if result:
                return result
    return None

# 4. DLS với heuristic
def dls_puzzle(current=None, limit=20, path=None, visited=None):
    """DLS cho 8-puzzle"""
    problem = PuzzleProblem()
    
    if current is None:
        current = problem.initial_state
    if path is None:
        path = [current]
    if visited is None:
        visited = set()
    
    if problem.is_goal(current):
        return path
    
    state_str = problem.state_to_string(current)
    if limit <= 0 or state_str in visited:
        return None
        
    visited.add(state_str)
    
    for action in problem.actions(current):
        next_state = problem.result(current, action)
        next_state_str = problem.state_to_string(next_state)
        
        if next_state_str not in visited:
            result = dls_puzzle(next_state, limit - 1, path + [next_state], visited.copy())
            if result:
                return result
    return None

# 5. IDS với heuristic
def ids_puzzle(max_depth=20):
    """IDS cho 8-puzzle"""
    for depth in range(max_depth + 1):
        result = dls_puzzle(limit=depth)
        if result:
            print(f"IDS tìm thấy ở độ sâu {depth}")
            return result
    return None
