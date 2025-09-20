"""
Traveling Salesman Problem (TSP) with BCO
Bài toán TSP thực tế - Tìm đường đi ngắn nhất qua tất cả các thành phố
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
from simple_bco import SimpleBCO


class TSProblem:
    """
    Traveling Salesman Problem
    """
    
    def __init__(self, cities: List[Tuple[float, float]], city_names: List[str] = None):
        """
        Khởi tạo TSP problem
        
        Args:
            cities: List các tọa độ thành phố [(x1, y1), (x2, y2), ...]
            city_names: Tên các thành phố (optional)
        """
        self.cities = np.array(cities)
        self.n_cities = len(cities)
        self.city_names = city_names or [f"City_{i+1}" for i in range(self.n_cities)]
        
        # Tính ma trận khoảng cách
        self.distance_matrix = self._calculate_distance_matrix()
        
        print(f"📍 TSP Problem với {self.n_cities} thành phố")
        print(f"Cities: {self.city_names}")
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        """
        Tính ma trận khoảng cách Euclidean giữa các thành phố
        """
        n = self.n_cities
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.cities[i][0] - self.cities[j][0]
                    dy = self.cities[i][1] - self.cities[j][1]
                    distances[i][j] = np.sqrt(dx*dx + dy*dy)
        
        return distances
    
    def calculate_tour_distance(self, tour: List[int]) -> float:
        """
        Tính tổng khoảng cách của một tour
        
        Args:
            tour: Danh sách thứ tự thành phố [0, 3, 1, 2, 0]
            
        Returns:
            Tổng khoảng cách của tour
        """
        total_distance = 0
        for i in range(len(tour) - 1):
            city_from = tour[i]
            city_to = tour[i + 1]
            total_distance += self.distance_matrix[city_from][city_to]
        
        return total_distance
    
    def visualize_tour(self, tour: List[int], title: str = "TSP Tour") -> None:
        """
        Vẽ tour TSP
        
        Args:
            tour: Thứ tự thành phố
            title: Tiêu đề biểu đồ
        """
        plt.figure(figsize=(10, 8))
        
        # Vẽ các thành phố
        x_coords = [self.cities[i][0] for i in range(self.n_cities)]
        y_coords = [self.cities[i][1] for i in range(self.n_cities)]
        
        plt.scatter(x_coords, y_coords, c='red', s=100, zorder=5)
        
        # Vẽ tour
        for i in range(len(tour) - 1):
            city_from = tour[i]
            city_to = tour[i + 1]
            
            x1, y1 = self.cities[city_from]
            x2, y2 = self.cities[city_to]
            
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
            
            # Vẽ mũi tên chỉ hướng
            dx, dy = x2 - x1, y2 - y1
            plt.arrow(x1 + 0.3*dx, y1 + 0.3*dy, 0.2*dx, 0.2*dy, 
                     head_width=0.3, head_length=0.2, fc='blue', ec='blue')
        
        # Ghi tên thành phố
        for i, (x, y) in enumerate(self.cities):
            plt.annotate(self.city_names[i], (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Tính và hiển thị tổng khoảng cách
        total_distance = self.calculate_tour_distance(tour)
        
        plt.title(f"{title}\nTotal Distance: {total_distance:.2f}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        return total_distance


class BCO_TSP(SimpleBCO):
    """
    BCO chuyên dụng cho TSP
    """
    
    def __init__(self, tsp_problem: TSProblem, **kwargs):
        """
        Khởi tạo BCO cho TSP
        
        Args:
            tsp_problem: Instance của TSProblem
        """
        super().__init__(**kwargs)
        self.tsp = tsp_problem
        self.n_cities = tsp_problem.n_cities
    
    def initialize_population(self, bounds=None) -> None:
        """
        Khởi tạo population cho TSP - mỗi solution là một tour
        """
        self.population = []
        
        for i in range(self.population_size):
            # Tạo tour ngẫu nhiên: bắt đầu và kết thúc tại city 0
            cities = list(range(1, self.n_cities))  # Exclude starting city 0
            random.shuffle(cities)
            tour = [0] + cities + [0]  # Start and end at city 0
            self.population.append(tour)
        
        self.trial_counters = np.zeros(self.population_size)
        print(f"✅ Step 01: Đã tạo {self.population_size} tours ngẫu nhiên")
    
    def calculate_fitness(self, objective_function=None) -> None:
        """
        Tính fitness cho TSP (distance càng nhỏ, fitness càng cao)
        """
        self.fitness = np.zeros(self.population_size)
        
        for i in range(self.population_size):
            distance = self.tsp.calculate_tour_distance(self.population[i])
            # Fitness = 1 / (1 + distance)
            self.fitness[i] = 1 / (1 + distance)
        
        print(f"✅ Step 02: Đã tính fitness cho {self.population_size} tours")
    
    def generate_new_tour_employed(self, i: int) -> List[int]:
        """
        Tạo tour mới cho employed bee bằng cách swap 2 cities
        """
        tour = self.population[i].copy()
        
        # Chọn 2 vị trí ngẫu nhiên (không bao gồm city đầu và cuối)
        pos1 = random.randint(1, len(tour) - 2)
        pos2 = random.randint(1, len(tour) - 2)
        
        # Swap 2 cities
        tour[pos1], tour[pos2] = tour[pos2], tour[pos1]
        
        return tour
    
    def employed_bee_phase(self, objective_function=None) -> None:
        """
        Employed bee phase cho TSP
        """
        print("🐝 Employed Bee Phase...")
        
        for i in range(self.population_size):
            # Tạo tour mới
            new_tour = self.generate_new_tour_employed(i)
            
            # Tính fitness của tour mới
            new_distance = self.tsp.calculate_tour_distance(new_tour)
            new_fitness = 1 / (1 + new_distance)
            
            # Greedy selection
            if new_fitness > self.fitness[i]:
                self.population[i] = new_tour
                self.fitness[i] = new_fitness
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
    
    def onlooker_bee_phase(self, objective_function=None) -> None:
        """
        Onlooker bee phase cho TSP
        """
        print("👁️ Onlooker Bee Phase...")
        
        # Tính probabilities
        probabilities = self.calculate_probabilities()
        
        for _ in range(self.population_size):
            # Chọn tour dựa trên probability
            i = self.roulette_wheel_selection(probabilities)
            
            # Tạo tour mới từ selected tour
            new_tour = self.generate_new_tour_employed(i)
            
            # Tính fitness
            new_distance = self.tsp.calculate_tour_distance(new_tour)
            new_fitness = 1 / (1 + new_distance)
            
            # Greedy selection
            if new_fitness > self.fitness[i]:
                self.population[i] = new_tour
                self.fitness[i] = new_fitness
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
    
    def scout_bee_phase(self, objective_function=None) -> None:
        """
        Scout bee phase cho TSP - tạo tour ngẫu nhiên mới
        """
        print("🔍 Scout Bee Phase...")
        
        abandoned_count = 0
        for i in range(self.population_size):
            if self.trial_counters[i] >= self.limit:
                # Tạo tour ngẫu nhiên mới
                cities = list(range(1, self.n_cities))
                random.shuffle(cities)
                new_tour = [0] + cities + [0]
                
                # Cập nhật
                self.population[i] = new_tour
                new_distance = self.tsp.calculate_tour_distance(new_tour)
                self.fitness[i] = 1 / (1 + new_distance)
                self.trial_counters[i] = 0
                abandoned_count += 1
        
        if abandoned_count > 0:
            print(f"   Đã abandon {abandoned_count} tours")
    
    def optimize_tsp(self, verbose: bool = True) -> Tuple[List[int], float]:
        """
        Tối ưu hóa TSP bằng BCO
        """
        print("🐝" * 20)
        print("BẮT ĐẦU BCO CHO TSP")
        print("🐝" * 20)
        
        # Step 01: Initialize population
        self.initialize_population()
        
        # Step 02: Calculate initial fitness
        self.calculate_fitness()
        
        # Step 03: Memorize best solution
        best_idx = np.argmax(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        
        print(f"✅ Step 03: Best initial distance = {self.tsp.calculate_tour_distance(self.best_solution):.2f}")
        
        # Lưu lịch sử
        self.distance_history = []
        
        print(f"\n📊 Bắt đầu optimization với {self.max_iterations} iterations...")
        
        # Main optimization loop
        for t in range(1, self.max_iterations + 1):
            if verbose and t % 10 == 0:
                current_distance = self.tsp.calculate_tour_distance(self.best_solution)
                print(f"\n--- Iteration {t} --- Best Distance: {current_distance:.2f}")
            
            # Employed bee phase
            self.employed_bee_phase()
            
            # Onlooker bee phase
            self.onlooker_bee_phase()
            
            # Scout bee phase
            self.scout_bee_phase()
            
            # Update best solution
            current_best_idx = np.argmax(self.fitness)
            if self.fitness[current_best_idx] > self.best_fitness:
                self.best_solution = self.population[current_best_idx].copy()
                self.best_fitness = self.fitness[current_best_idx]
                if verbose:
                    new_distance = self.tsp.calculate_tour_distance(self.best_solution)
                    print(f"   ✨ Tìm thấy tour tốt hơn! Distance = {new_distance:.2f}")
            
            # Lưu lịch sử
            best_distance = self.tsp.calculate_tour_distance(self.best_solution)
            self.distance_history.append(best_distance)
        
        final_distance = self.tsp.calculate_tour_distance(self.best_solution)
        
        print(f"\n🎉 BCO HOÀN THÀNH!")
        print(f"Best tour: {self.best_solution}")
        print(f"Best distance: {final_distance:.2f}")
        
        return self.best_solution, final_distance
    
    def plot_convergence(self) -> None:
        """
        Vẽ biểu đồ hội tụ cho TSP
        """
        if not hasattr(self, 'distance_history') or not self.distance_history:
            print("Chưa có dữ liệu để vẽ!")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.distance_history, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Best Distance')
        plt.title('BCO Convergence for TSP - Best Distance Evolution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def create_vietnam_cities():
    """
    Tạo bài toán TSP với các thành phố Việt Nam (tọa độ giả định)
    """
    cities = [
        (10.8, 106.7),   # TP.HCM (Sài Gòn)
        (21.0, 105.8),   # Hà Nội
        (16.5, 107.6),   # Huế
        (15.9, 108.3),   # Đà Nẵng
        (12.2, 109.2),   # Nha Trang
        (10.0, 105.8),   # Cần Thơ
        (20.9, 106.9),   # Hải Phòng
        (18.3, 105.9),   # Vinh
    ]
    
    city_names = [
        "TP.HCM", "Hà Nội", "Huế", "Đà Nẵng", 
        "Nha Trang", "Cần Thơ", "Hải Phòng", "Vinh"
    ]
    
    return cities, city_names


def demo_tsp_simple():
    """
    Demo BCO cho TSP với một bài toán nhỏ
    """
    print("="*60)
    print("DEMO BCO CHO TSP - CÁC THÀNH PHỐ VIỆT NAM")
    print("="*60)
    
    # Tạo bài toán TSP
    cities, city_names = create_vietnam_cities()
    tsp = TSProblem(cities, city_names)
    
    # Hiển thị ma trận khoảng cách
    print(f"\n📊 Ma trận khoảng cách (đơn giản hóa):")
    print("     ", end="")
    for name in city_names:
        print(f"{name[:6]:>8}", end="")
    print()
    
    for i, name in enumerate(city_names):
        print(f"{name[:6]:>6}: ", end="")
        for j in range(len(cities)):
            print(f"{tsp.distance_matrix[i][j]:>8.1f}", end="")
        print()
    
    # Tạo BCO solver
    bco_tsp = BCO_TSP(
        tsp_problem=tsp,
        population_size=20,
        max_iterations=100,
        limit=8
    )
    
    print(f"\n🎯 Mục tiêu: Tìm tour ngắn nhất đi qua tất cả {len(cities)} thành phố")
    print("Bắt đầu và kết thúc tại TP.HCM")
    
    # Tạo tour ngẫu nhiên để so sánh
    random_tour = [0] + list(range(1, len(cities))) + [0]
    random.shuffle(random_tour[1:-1])  # Shuffle middle cities
    random_distance = tsp.calculate_tour_distance(random_tour)
    
    print(f"\n🎲 Tour ngẫu nhiên:")
    tour_str = " → ".join([city_names[i] for i in random_tour])
    print(f"   {tour_str}")
    print(f"   Distance: {random_distance:.2f}")
    
    # Chạy BCO optimization
    best_tour, best_distance = bco_tsp.optimize_tsp(verbose=True)
    
    # So sánh kết quả
    improvement = ((random_distance - best_distance) / random_distance) * 100
    
    print(f"\n📋 KẾT QUẢ SO SÁNH:")
    print(f"Tour ngẫu nhiên: {random_distance:.2f}")
    print(f"Tour BCO:        {best_distance:.2f}")
    print(f"Cải thiện:       {improvement:.1f}%")
    
    if improvement > 20:
        print("🎉 EXCELLENT: BCO cải thiện đáng kể!")
    elif improvement > 10:
        print("✅ GOOD: BCO có cải thiện tốt!")
    elif improvement > 0:
        print("⚠️ OK: BCO có cải thiện nhẹ!")
    else:
        print("❌ POOR: BCO cần điều chỉnh parameters!")
    
    # Visualization
    print(f"\n🗺️ Hiển thị tours...")
    
    # Vẽ tour ngẫu nhiên
    tsp.visualize_tour(random_tour, "Random Tour")
    
    # Vẽ tour BCO
    tsp.visualize_tour(best_tour, "BCO Optimized Tour")
    
    # Vẽ convergence
    bco_tsp.plot_convergence()
    
    return best_tour, best_distance


if __name__ == "__main__":
    # Chạy demo TSP
    demo_tsp_simple()