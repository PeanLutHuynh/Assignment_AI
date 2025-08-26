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

# Các thuật toán với HEURISTIC LINH HOẠT cho HelloProblem

def is_promising_path(state):
    """Relaxed heuristic - cho phép sai vài ký tự"""
    if len(state) > len(GOAL):
        return False
    if len(state) == 0:
        return True
    
    # Tính số ký tự khác nhau
    differences = sum(1 for i, char in enumerate(state) 
                     if i < len(GOAL) and char != GOAL[i])

    # Cho phép sai tối đa 2 ký tự hoặc 20%
    max_errors = max(1, min(2, len(state) // 5))
    return differences <= max_errors

# 1. BFS với heuristic linh hoạt
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

        if nodes_explored > 1000000:
            break
            
        for action in problem.actions(current):
            next_state = problem.result(current, action)
            
            if next_state not in visited:
                # Ưu tiên đường đúng, nhưng vẫn thử đường khác
                if is_promising_path(next_state):
                    new_path = path + [next_state]
                    
                    if problem.is_goal(next_state):
                        return new_path
                    
                    queue.append((next_state, new_path))
                    visited.add(next_state)
    return None

# 2. UCS với heuristic linh hoạt
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
        
        # Tăng giới hạn
        if nodes_explored > 1000000:
            break
            
        if current in visited:
            continue
        visited.add(current)
        
        if problem.is_goal(current):
            return path, cost
        
        for action in problem.actions(current):
            next_state = problem.result(current, action)
                
            if next_state not in visited:
                # Ưu tiên đường đúng nhưng cho phép khác
                if is_promising_path(next_state):
                    new_cost = cost + problem.get_cost(current, action)
                    new_path = path + [next_state]
                    heapq.heappush(heap, (new_cost, next_state, new_path))
    
    return None, float('inf')

# 3. DFS với heuristic linh hoạt
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
    
    # Ưu tiên đường đúng, nhưng vẫn thử đường khác
    if not is_promising_path(current):
        return None
        
    visited.add(current)
    
    for action in problem.actions(current):
        next_state = problem.result(current, action)
        if next_state not in visited:
            if is_promising_path(next_state):
                result = dfs_hello(next_state, path + [next_state], visited.copy(), depth + 1, max_depth)
                if result:
                    return result
    return None

# 4. DLS với heuristic linh hoạt
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
    
    # Ưu tiên đường đúng nhưng cho phép thử khác
    if not is_promising_path(current):
        return None
        
    visited.add(current)
    
    for action in problem.actions(current):
        next_state = problem.result(current, action)
        if next_state not in visited:
            if is_promising_path(next_state):
                result = dls_hello(next_state, limit - 1, path + [next_state], visited.copy())
                if result:
                    return result
    return None

# 5. IDS với heuristic linh hoạt
def ids_hello(max_depth=11):
    for depth in range(max_depth + 1):
        result = dls_hello('', depth)
        if result:
            return result
    return None
