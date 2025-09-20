"""
Simple Bee Colony Optimization (BCO) Implementation
Theo đúng flowchart chuẩn - Đơn giản và dễ hiểu
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable


class SimpleBCO:
    """
    Thuật toán BCO đơn giản theo flowchart chuẩn
    """
    
    def __init__(self, 
                 population_size: int = 20,
                 max_iterations: int = 100,
                 limit: int = 10):
        """
        Khởi tạo BCO
        
        Args:
            population_size: Kích thước quần thể (số food sources)
            max_iterations: Số vòng lặp tối đa (MaxT)
            limit: Giới hạn abandon (số lần thử không cải thiện)
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.limit = limit
        
        # Lưu trữ quần thể và thông tin
        self.population = None      # Xi - các food sources
        self.fitness = None         # Fitness của mỗi solution
        self.trial_counters = None  # Đếm số lần thử cho mỗi solution
        self.best_solution = None   # XBest - best solution
        self.best_fitness = None    # Fitness của best solution
        
        # Lịch sử optimization
        self.fitness_history = []
        
    def initialize_population(self, bounds: List[Tuple[float, float]]) -> None:
        """
        Step 01: Generate initial population randomly (Xi), i = 1,2,3,4,...Population Size
        """
        self.dimension = len(bounds)
        self.bounds = bounds
        
        # Tạo population ngẫu nhiên
        self.population = np.zeros((self.population_size, self.dimension))
        
        for i in range(self.population_size):
            for j in range(self.dimension):
                lower, upper = bounds[j]
                self.population[i, j] = random.uniform(lower, upper)
        
        # Khởi tạo trial counters
        self.trial_counters = np.zeros(self.population_size)
        
        print(f"✅ Step 01: Đã tạo {self.population_size} solutions ngẫu nhiên")
    
    def calculate_fitness(self, objective_function: Callable) -> None:
        """
        Step 02: Calculate fitness values for each agent in the population
        """
        self.fitness = np.zeros(self.population_size)
        
        for i in range(self.population_size):
            cost = objective_function(self.population[i])
            # Chuyển cost thành fitness (fitness càng cao càng tốt)
            self.fitness[i] = 1 / (1 + cost) if cost >= 0 else 1 + abs(cost)
        
        print(f"✅ Step 02: Đã tính fitness cho {self.population_size} solutions")
    
    def memorize_best_solution(self) -> None:
        """
        Step 03: Memorize the best (XBest) solution in the population
        """
        best_idx = np.argmax(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        
        print(f"✅ Step 03: Best fitness = {self.best_fitness:.6f}")
    
    def generate_new_solution_employed(self, i: int) -> np.ndarray:
        """
        Step 05: Generate new solutions for employee bee (vi) from old solutions (Xi)
        
        Công thức: vij = xij + φij(xij - xkj)
        """
        # Chọn ngẫu nhiên một solution khác (k ≠ i)
        k = random.randint(0, self.population_size - 1)
        while k == i:
            k = random.randint(0, self.population_size - 1)
        
        # Chọn ngẫu nhiên một dimension để thay đổi
        j = random.randint(0, self.dimension - 1)
        
        # Tạo solution mới
        new_solution = self.population[i].copy()
        phi = random.uniform(-1, 1)  # φij
        new_solution[j] = self.population[i, j] + phi * (self.population[i, j] - self.population[k, j])
        
        # Đảm bảo trong bounds
        lower, upper = self.bounds[j]
        new_solution[j] = np.clip(new_solution[j], lower, upper)
        
        return new_solution
    
    def employed_bee_phase(self, objective_function: Callable) -> None:
        """
        Step 05-07: Employed bee phase
        """
        print("🐝 Employed Bee Phase...")
        
        for i in range(self.population_size):
            # Step 05: Generate new solution
            new_solution = self.generate_new_solution_employed(i)
            
            # Step 06: Compute fitness of new solution
            new_cost = objective_function(new_solution)
            new_fitness = 1 / (1 + new_cost) if new_cost >= 0 else 1 + abs(new_cost)
            
            # Step 07: Keep the best solution between current and candidate
            if new_fitness > self.fitness[i]:  # Greedy selection
                self.population[i] = new_solution
                self.fitness[i] = new_fitness
                self.trial_counters[i] = 0  # Reset counter
            else:
                self.trial_counters[i] += 1  # Increment counter
    
    def calculate_probabilities(self) -> np.ndarray:
        """
        Step 08: Calculate the Probability (Pi) for the solution (Xi)
        
        Pi = fitnessi / Σ(fitness)
        """
        total_fitness = np.sum(self.fitness)
        if total_fitness == 0:
            return np.ones(self.population_size) / self.population_size
        
        probabilities = self.fitness / total_fitness
        return probabilities
    
    def onlooker_bee_phase(self, objective_function: Callable) -> None:
        """
        Step 09-10: Onlooker bee phase
        """
        print("👁️ Onlooker Bee Phase...")
        
        # Step 08: Calculate probabilities
        probabilities = self.calculate_probabilities()
        
        # Step 09: Generate new solutions for onlooker bees
        for _ in range(self.population_size):  # Mỗi onlooker bee chọn một solution
            # Roulette wheel selection dựa trên probability
            i = self.roulette_wheel_selection(probabilities)
            
            # Generate new solution từ selected solution
            new_solution = self.generate_new_solution_employed(i)
            
            # Step 10: Calculate fitness of new solution
            new_cost = objective_function(new_solution)
            new_fitness = 1 / (1 + new_cost) if new_cost >= 0 else 1 + abs(new_cost)
            
            # Greedy selection
            if new_fitness > self.fitness[i]:
                self.population[i] = new_solution
                self.fitness[i] = new_fitness
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
    
    def roulette_wheel_selection(self, probabilities: np.ndarray) -> int:
        """
        Roulette wheel selection dựa trên probabilities
        """
        cumsum = np.cumsum(probabilities)
        r = random.random()
        
        for i, cum_prob in enumerate(cumsum):
            if r <= cum_prob:
                return i
        
        return len(probabilities) - 1
    
    def scout_bee_phase(self, objective_function: Callable) -> None:
        """
        Step 11: Determine the abandoned solution if exist, replace it with new randomly solution Xi
        """
        print("🔍 Scout Bee Phase...")
        
        abandoned_count = 0
        for i in range(self.population_size):
            if self.trial_counters[i] >= self.limit:
                # Abandon solution và tạo random solution mới
                for j in range(self.dimension):
                    lower, upper = self.bounds[j]
                    self.population[i, j] = random.uniform(lower, upper)
                
                # Tính fitness cho solution mới
                new_cost = objective_function(self.population[i])
                self.fitness[i] = 1 / (1 + new_cost) if new_cost >= 0 else 1 + abs(new_cost)
                self.trial_counters[i] = 0
                abandoned_count += 1
        
        if abandoned_count > 0:
            print(f"   Đã abandon {abandoned_count} solutions")
    
    def update_best_solution(self) -> None:
        """
        Step 12: Keep the best solution found in the population
        """
        current_best_idx = np.argmax(self.fitness)
        current_best_fitness = self.fitness[current_best_idx]
        
        if current_best_fitness > self.best_fitness:
            self.best_solution = self.population[current_best_idx].copy()
            self.best_fitness = current_best_fitness
            print(f"   ✨ Tìm thấy solution tốt hơn! Fitness = {self.best_fitness:.6f}")
    
    def optimize(self, 
                objective_function: Callable,
                bounds: List[Tuple[float, float]],
                verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Chạy thuật toán BCO hoàn chỉnh theo flowchart
        """
        print("🐝" * 20)
        print("BẮT ĐẦU THUẬT TOÁN BCO")
        print("🐝" * 20)
        
        # Step 01: Initialize population
        self.initialize_population(bounds)
        
        # Step 02: Calculate initial fitness
        self.calculate_fitness(objective_function)
        
        # Step 03: Memorize best solution
        self.memorize_best_solution()
        
        # Step 04: Set current iteration (t = 1)
        self.fitness_history = []
        
        print(f"\n📊 Bắt đầu optimization với {self.max_iterations} iterations...")
        
        # Main loop: Step 13-14
        for t in range(1, self.max_iterations + 1):
            if verbose and t % 20 == 0:
                print(f"\n--- Iteration {t} ---")
            
            # Step 05-07: Employed bee phase
            self.employed_bee_phase(objective_function)
            
            # Step 08-10: Onlooker bee phase  
            self.onlooker_bee_phase(objective_function)
            
            # Step 11: Scout bee phase
            self.scout_bee_phase(objective_function)
            
            # Step 12: Update best solution
            self.update_best_solution()
            
            # Lưu lịch sử
            self.fitness_history.append(self.best_fitness)
            
            # Step 13: t = t+1 (automatic in for loop)
            # Step 14: Repeat until t <= MaxT (automatic in for loop)
        
        # Chuyển fitness thành cost để return
        best_cost = (1 / self.best_fitness) - 1 if self.best_fitness > 0 else -(self.best_fitness - 1)
        
        print(f"\n🎉 BCO HOÀN THÀNH!")
        print(f"Best solution: {self.best_solution}")
        print(f"Best cost: {best_cost:.6f}")
        
        return self.best_solution, best_cost
    
    def plot_convergence(self) -> None:
        """
        Vẽ biểu đồ hội tụ
        """
        if not self.fitness_history:
            print("Chưa có dữ liệu để vẽ!")
            return
        
        # Chuyển fitness về cost để vẽ
        costs = [(1/f) - 1 if f > 0 else -(f-1) for f in self.fitness_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(costs, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Best Cost')
        plt.title('BCO Convergence - Best Cost Evolution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def test_simple_function():
    """
    Test BCO với hàm đơn giản: f(x,y) = x² + y²
    """
    print("="*50)
    print("TEST BCO VỚI HÀM ĐƠN GIẢN")
    print("="*50)
    
    def sphere_function(x):
        """Hàm Sphere: f(x) = x₁² + x₂² + ... + xₙ²"""
        return np.sum(x**2)
    
    # Khởi tạo BCO
    bco = SimpleBCO(
        population_size=15,
        max_iterations=80,
        limit=5
    )
    
    # Định nghĩa bounds cho bài toán 2D
    bounds = [(-5, 5), (-5, 5)]  # x ∈ [-5, 5], y ∈ [-5, 5]
    
    print(f"Bài toán: Minimize f(x,y) = x² + y²")
    print(f"Domain: x ∈ [-5, 5], y ∈ [-5, 5]")
    print(f"Global minimum: f(0, 0) = 0")
    
    # Chạy optimization
    best_solution, best_cost = bco.optimize(sphere_function, bounds, verbose=True)
    
    # Đánh giá kết quả
    distance_from_optimum = np.sqrt(best_solution[0]**2 + best_solution[1]**2)
    
    print(f"\n📋 ĐÁNH GIÁ KẾT QUẢ:")
    print(f"Khoảng cách từ global optimum (0,0): {distance_from_optimum:.4f}")
    
    if best_cost < 0.01:
        print("🎉 EXCELLENT: BCO tìm được solution rất gần global optimum!")
    elif best_cost < 0.1:
        print("✅ GOOD: BCO tìm được solution khá tốt!")
    elif best_cost < 1.0:
        print("⚠️ OK: BCO tìm được solution acceptable!")
    else:
        print("❌ POOR: BCO cần tune parameters!")
    
    # Vẽ convergence
    bco.plot_convergence()
    
    return best_solution, best_cost


if __name__ == "__main__":
    # Chạy test đơn giản
    test_simple_function()
