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
            # Initial state:
            self.initial_state = [
                [1, 4, 0],
                [5, 2, 3],
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

# Heuristic linh hoạt cho 8-puzzle
def is_promising_state(state):
    """Kiểm tra xem state có triển vọng dẫn đến goal không"""
    # Manhattan distance không quá 30 (khoảng cách hợp lý)
    distance = manhattan_distance(state)
    return distance <= 30

def get_heuristic_priority(state):
    """Trả về độ ưu tiên dựa trên heuristic (càng nhỏ càng tốt)"""
    return manhattan_distance(state)

def find_goal_position(value):
    """Tìm vị trí của value trong goal state"""
    for i in range(3):
        for j in range(3):
            if GOAL_STATE[i][j] == value:
                return i, j
    return None


# Các thuật toán TỐI ƯU với HEURISTIC cho 8-puzzle
# 1. BFS với heuristic linh hoạt
def bfs_puzzle():
    """BFS cho 8-puzzle với heuristic guidance"""
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
        
        # Tăng giới hạn để tìm được nghiệm
        if nodes_explored > 100000:
            break
            
        for action in problem.actions(current):
            next_state = problem.result(current, action)
            state_str = problem.state_to_string(next_state)
            
            if state_str not in visited:
                # Sử dụng heuristic để ưu tiên nhưng không loại bỏ hoàn toàn
                if is_promising_state(next_state):
                    new_path = path + [next_state]
                    
                    if problem.is_goal(next_state):
                        return new_path
                    
                    queue.append((next_state, new_path))
                    visited.add(state_str)
    return None

# 2. UCS với heuristic linh hoạt
def ucs_puzzle():
    """UCS cho 8-puzzle với heuristic guidance"""
    problem = PuzzleProblem()
    start = problem.initial_state
    
    if problem.is_goal(start):
        return [start], 0
    
    # Sử dụng heuristic để sắp xếp thứ tự ưu tiên
    heap = [(get_heuristic_priority(start), 0, problem.state_to_string(start), start, [start])]
    visited = set()
    nodes_explored = 0
    
    while heap:
        _, cost, state_str, current, path = heapq.heappop(heap)
        nodes_explored += 1
        
        # Tăng giới hạn
        if nodes_explored > 100000:
            break
            
        if state_str in visited:
            continue
        visited.add(state_str)
        
        if problem.is_goal(current):
            return path, cost
        
        for action in problem.actions(current):
            next_state = problem.result(current, action)
            next_state_str = problem.state_to_string(next_state)
            
            if next_state_str not in visited and is_promising_state(next_state):
                new_cost = cost + problem.get_cost(current, action)
                new_path = path + [next_state]
                priority = new_cost + get_heuristic_priority(next_state) * 0.5  # A* style
                heapq.heappush(heap, (priority, new_cost, next_state_str, next_state, new_path))
    
    return None, float('inf')

# 3. DFS với heuristic linh hoạt
def dfs_puzzle(current=None, path=None, visited=None, depth=0, max_depth=20):
    """DFS cho 8-puzzle với heuristic guidance"""
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
    
    # Sắp xếp actions theo heuristic để ưu tiên đường tốt hơn
    actions = problem.actions(current)
    action_states = []
    
    for action in actions:
        next_state = problem.result(current, action)
        if is_promising_state(next_state):
            priority = get_heuristic_priority(next_state)
            action_states.append((priority, action, next_state))
    
    # Sắp xếp theo heuristic (priority thấp = tốt hơn)
    action_states.sort(key=lambda x: x[0])
    
    for _, action, next_state in action_states:
        next_state_str = problem.state_to_string(next_state)
        
        if next_state_str not in visited:
            result = dfs_puzzle(next_state, path + [next_state], visited.copy(), depth + 1, max_depth)
            if result:
                return result
    return None

# 4. DLS với heuristic linh hoạt
def dls_puzzle(current=None, limit=20, path=None, visited=None):
    """DLS cho 8-puzzle với heuristic guidance"""
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
    
    # Sắp xếp actions theo heuristic
    actions = problem.actions(current)
    action_states = []
    
    for action in actions:
        next_state = problem.result(current, action)
        if is_promising_state(next_state):
            priority = get_heuristic_priority(next_state)
            action_states.append((priority, action, next_state))
    
    action_states.sort(key=lambda x: x[0])
    
    for _, action, next_state in action_states:
        next_state_str = problem.state_to_string(next_state)
        
        if next_state_str not in visited:
            result = dls_puzzle(next_state, limit - 1, path + [next_state], visited.copy())
            if result:
                return result
    return None

# 5. IDS với heuristic linh hoạt
def ids_puzzle(max_depth=20):
    """IDS cho 8-puzzle với heuristic guidance"""
    for depth in range(max_depth + 1):
        result = dls_puzzle(limit=depth)
        if result:
            return result
    return None
