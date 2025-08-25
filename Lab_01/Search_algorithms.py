# Bài 1/ Các thuật toán tìm kiếm cơ bản

from collections import deque
import heapq
from typing import List, Dict, Tuple, Optional, Set

class Graph:
    """Class để biểu diễn đồ thị"""
    def __init__(self):
        self.graph = {}
        self.costs = {}
    
    def add_edge(self, u, v, cost=1):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
        self.costs[(u, v)] = cost
    
    def get_neighbors(self, node):
        return self.graph.get(node, [])
    
    def get_cost(self, from_node, to_node):
        return self.costs.get((from_node, to_node), 1)

# 1. BFS
def bfs(graph: Graph, start, goal):
    if start == goal:
        return [start]
    
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        current, path = queue.popleft()
        
        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == goal:
                    return new_path
                queue.append((neighbor, new_path))
                visited.add(neighbor)
    return None

# 2. UCS - tái sử dụng BFS
def ucs(graph: Graph, start, goal):
    if start == goal:
        return [start], 0
    
    heap = [(0, start, [start])]
    visited = set()
    
    while heap:
        cost, current, path = heapq.heappop(heap)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == goal:
            return path, cost
        
        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                new_cost = cost + graph.get_cost(current, neighbor)
                new_path = path + [neighbor]
                heapq.heappush(heap, (new_cost, neighbor, new_path))
    
    return None, float('inf')

# 3. DFS
def dfs(graph: Graph, start, goal, visited=None):
    if visited is None:
        visited = set()
    
    if start == goal:
        return [start]
    
    visited.add(start)
    
    for neighbor in graph.get_neighbors(start):
        if neighbor not in visited:
            path = dfs(graph, neighbor, goal, visited.copy())
            if path:
                return [start] + path
    return None

# 4. DLS - tái sử dụng DFS
def dls(graph: Graph, start, goal, limit, visited=None):
    if visited is None:
        visited = set()
    
    if start == goal:
        return [start]
    
    if limit <= 0:
        return None
    
    visited.add(start)
    
    for neighbor in graph.get_neighbors(start):
        if neighbor not in visited:
            path = dls(graph, neighbor, goal, limit - 1, visited.copy())
            if path:
                return [start] + path
    return None

# 5. IDS - tái sử dụng DLS
def ids(graph: Graph, start, goal, max_depth=100):
    for depth in range(max_depth + 1):
        result = dls(graph, start, goal, depth)
        if result:
            return result
    return None
