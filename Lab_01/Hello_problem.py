# Bài 2/ Sinh string ngẫu nhiên => HELLO WORLD

from collections import deque
import heapq

GOAL = 'HELLO WORLD'

class HelloProblem:
    
    def __init__(self, initial_state=''):
        self.initial_state = initial_state
    
    def actions(self, state):
        if len(state) < len(GOAL):
            return list(' ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        return []
    
    def result(self, state, action):
        return state + action
    
    def is_goal(self, state):
        return state == GOAL
    
    def get_cost(self, state, action):
        return 1

# Các thuật toán TỐI ƯU với HEURISTIC cho HelloProblem

def is_on_correct_path(state):
    """Kiểm tra xem state có đang đi đúng hướng HELLO WORLD không"""
    if len(state) > len(GOAL):
        return False
    return GOAL.startswith(state)

# 1. BFS với heuristic
def bfs_hello():
    problem = HelloProblem()
    start = problem.initial_state
    
    if start == GOAL:
        return [start]
    
    queue = deque([(start, [start])])
    visited = set([start])
    nodes_explored = 0
    
    while queue:
        current, path = queue.popleft()
        nodes_explored += 1
        
        # Early stopping nếu quá nhiều nodes
        if nodes_explored > 100000:
            print(f"BFS dừng sớm sau {nodes_explored} nodes")
            break
            
        for action in problem.actions(current):
            next_state = problem.result(current, action)
            
            # HEURISTIC: Chỉ tiếp tục nếu đi đúng hướng
            if not is_on_correct_path(next_state):
                continue
                
            if next_state not in visited:
                new_path = path + [next_state]
                
                if problem.is_goal(next_state):
                    return new_path
                
                queue.append((next_state, new_path))
                visited.add(next_state)
    return None

# 2. UCS với heuristic
def ucs_hello():
    problem = HelloProblem()
    start = problem.initial_state
    
    if problem.is_goal(start):
        return [start], 0
    
    heap = [(0, start, [start])]
    visited = set()
    nodes_explored = 0
    
    while heap:
        cost, current, path = heapq.heappop(heap)
        nodes_explored += 1
        
        # Early stopping
        if nodes_explored > 100000:
            print(f"UCS dừng sớm sau {nodes_explored} nodes")
            break
            
        if current in visited:
            continue
        visited.add(current)
        
        if problem.is_goal(current):
            return path, cost
        
        for action in problem.actions(current):
            next_state = problem.result(current, action)
            
            # HEURISTIC: Chỉ tiếp tục nếu đi đúng hướng  
            if not is_on_correct_path(next_state):
                continue
                
            if next_state not in visited:
                new_cost = cost + problem.get_cost(current, action)
                new_path = path + [next_state]
                heapq.heappush(heap, (new_cost, next_state, new_path))
    
    return None, float('inf')

# 3. DFS với heuristic và timeout
def dfs_hello(current='', path=None, visited=None, depth=0, max_depth=11):
    problem = HelloProblem()
    
    if path is None:
        path = [current]
    if visited is None:
        visited = set()
    
    if problem.is_goal(current):
        return path
    
    # Giới hạn độ sâu để tránh chạy mãi
    if depth >= max_depth or current in visited:
        return None
    
    # HEURISTIC: Chỉ tiếp tục nếu đi đúng hướng
    if not is_on_correct_path(current):
        return None
        
    visited.add(current)
    
    for action in problem.actions(current):
        next_state = problem.result(current, action)
        if next_state not in visited and is_on_correct_path(next_state):
            result = dfs_hello(next_state, path + [next_state], visited.copy(), depth + 1, max_depth)
            if result:
                return result
    return None

# 4. DLS với heuristic
def dls_hello(current='', limit=11, path=None, visited=None):
    problem = HelloProblem()
    
    if path is None:
        path = [current]
    if visited is None:
        visited = set()
    
    if problem.is_goal(current):
        return path
    
    if limit <= 0 or current in visited:
        return None
    
    # HEURISTIC: Chỉ tiếp tục nếu đi đúng hướng
    if not is_on_correct_path(current):
        return None
        
    visited.add(current)
    
    for action in problem.actions(current):
        next_state = problem.result(current, action)
        if next_state not in visited and is_on_correct_path(next_state):
            result = dls_hello(next_state, limit - 1, path + [next_state], visited.copy())
            if result:
                return result
    return None

# 5. IDS với heuristic
def ids_hello(max_depth=11):
    for depth in range(max_depth + 1):
        result = dls_hello('', depth)
        if result:
            print(f"IDS tìm thấy ở độ sâu {depth}")
            return result
    return None
