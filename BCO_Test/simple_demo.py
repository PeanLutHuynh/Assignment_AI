"""
Demo tổng hợp BCO đơn giản
Hiểu cách BCO hoạt động qua 2 bài toán: Function Optimization và TSP
"""

from simple_bco import SimpleBCO, test_simple_function
from tsp_bco import demo_tsp_simple, create_vietnam_cities, TSProblem, BCO_TSP
import numpy as np


def demo_bco_step_by_step():
    """
    Demo từng bước của BCO để hiểu thuật toán
    """
    print("🐝" * 30)
    print("DEMO BCO STEP-BY-STEP")
    print("🐝" * 30)
    
    print("\n📚 GIẢI THÍCH THUẬT TOÁN BCO:")
    print("BCO mô phỏng hành vi tìm kiếm thức ăn của đàn ong:")
    print("1️⃣  EMPLOYED BEES: Khai thác các nguồn thức ăn hiện tại")
    print("2️⃣  ONLOOKER BEES: Chọn nguồn thức ăn tốt dựa trên thông tin")
    print("3️⃣  SCOUT BEES: Tìm kiếm nguồn thức ăn mới khi cũ không tốt")
    
    print("\n" + "="*50)
    print("FLOWCHART BCO:")
    print("="*50)
    print("Step 01: Generate initial population randomly (Xi)")
    print("Step 02: Calculate fitness values for each agent")
    print("Step 03: Memorize the best (XBest) solution")
    print("Step 04: Set Current Iteration (t = 1)")
    print("Step 05: Generate new solutions for employee bee (vi)")
    print("Step 06: Compute fitness of all new solutions")
    print("Step 07: Keep the best solution (greedy selection)")
    print("Step 08: Calculate the Probability (Pi) for each solution")
    print("Step 09: Generate new solutions for onlooker bees")
    print("Step 10: Calculate fitness of all new solutions")
    print("Step 11: Replace abandoned solutions with random ones")
    print("Step 12: Keep the best solution found")
    print("Step 13: t = t+1")
    print("Step 14: Repeat until t <= MaxT")
    
    input("\n⏸️  Nhấn Enter để tiếp tục demo...")


def demo_function_optimization():
    """
    Demo BCO cho function optimization
    """
    print("\n" + "🔢" * 30)
    print("DEMO 1: FUNCTION OPTIMIZATION")
    print("🔢" * 30)
    
    print("\n📝 Bài toán: Minimize f(x,y) = x² + y²")
    print("🎯 Mục tiêu: Tìm (x,y) sao cho f(x,y) nhỏ nhất")
    print("🔍 Global minimum: f(0,0) = 0")
    
    input("⏸️  Nhấn Enter để chạy BCO...")
    
    # Chạy demo function optimization
    best_solution, best_cost = test_simple_function()
    
    print(f"\n📊 PHÂN TÍCH KẾT QUẢ:")
    print(f"• BCO tìm được solution: ({best_solution[0]:.4f}, {best_solution[1]:.4f})")
    print(f"• Cost: {best_cost:.6f}")
    print(f"• Distance from optimum: {np.sqrt(best_solution[0]**2 + best_solution[1]**2):.4f}")
    
    return best_solution, best_cost


def demo_tsp_explanation():
    """
    Demo BCO cho TSP với giải thích chi tiết
    """
    print("\n" + "🗺️" * 30)
    print("DEMO 2: TRAVELING SALESMAN PROBLEM (TSP)")
    print("🗺️" * 30)
    
    print("\n📝 Bài toán TSP:")
    print("• Có N thành phố cần đi qua")
    print("• Bắt đầu từ thành phố A, đi qua tất cả thành phố khác đúng 1 lần")
    print("• Quay về thành phố A")
    print("• Tìm tour có tổng khoảng cách ngắn nhất")
    
    print("\n🧠 Cách BCO giải TSP:")
    print("• Mỗi 'food source' = một tour (thứ tự đi qua các thành phố)")
    print("• Employed bees: Thay đổi tour bằng cách swap 2 thành phố")
    print("• Onlooker bees: Chọn tour tốt để cải thiện")
    print("• Scout bees: Tạo tour ngẫu nhiên mới khi tour cũ không cải thiện")
    
    input("⏸️  Nhấn Enter để chạy BCO cho TSP...")
    
    # Chạy demo TSP
    best_tour, best_distance = demo_tsp_simple()
    
    print(f"\n📊 PHÂN TÍCH TSP:")
    print(f"• Tour tốt nhất có {len(best_tour)} thành phố")
    print(f"• Tổng khoảng cách: {best_distance:.2f}")
    print(f"• BCO đã tìm được tour tối ưu hơn tour ngẫu nhiên")
    
    return best_tour, best_distance


def compare_bco_parameters():
    """
    So sánh các tham số khác nhau của BCO
    """
    print("\n" + "⚙️" * 30)
    print("DEMO 3: PARAMETER TUNING")
    print("⚙️" * 30)
    
    print("\n🔧 Các tham số quan trọng của BCO:")
    print("• Population Size: Số lượng food sources")
    print("• Max Iterations: Số vòng lặp tối đa")
    print("• Limit: Số lần thử trước khi abandon solution")
    
    # Test với các population size khác nhau
    print("\n📊 Test với Population Size khác nhau:")
    
    def simple_sphere(x):
        return x[0]**2 + x[1]**2
    
    bounds = [(-5, 5), (-5, 5)]
    pop_sizes = [5, 10, 20, 30]
    results = []
    
    for pop_size in pop_sizes:
        print(f"\n🧪 Testing Population Size = {pop_size}...")
        
        bco = SimpleBCO(
            population_size=pop_size,
            max_iterations=50,
            limit=5
        )
        
        solution, cost = bco.optimize(simple_sphere, bounds, verbose=False)
        results.append((pop_size, cost))
        
        print(f"   Result: Cost = {cost:.6f}")
    
    print(f"\n📈 COMPARISON RESULTS:")
    print("Population Size | Best Cost")
    print("-" * 25)
    for pop_size, cost in results:
        print(f"{pop_size:>12} | {cost:>9.6f}")
    
    # Tìm best result
    best_result = min(results, key=lambda x: x[1])
    print(f"\n🏆 Best Population Size: {best_result[0]} (Cost: {best_result[1]:.6f})")


def interactive_demo():
    """
    Demo tương tác cho người dùng
    """
    print("\n" + "🎮" * 30)
    print("INTERACTIVE BCO DEMO")
    print("🎮" * 30)
    
    while True:
        print("\n🐝 Chọn demo muốn xem:")
        print("1. Giải thích BCO step-by-step")
        print("2. Function Optimization (x² + y²)")
        print("3. TSP - Traveling Salesman Problem")
        print("4. Parameter Tuning")
        print("5. Tạo TSP problem tùy chỉnh")
        print("0. Thoát")
        
        choice = input("\nNhập lựa chọn (0-5): ").strip()
        
        if choice == '1':
            demo_bco_step_by_step()
        elif choice == '2':
            demo_function_optimization()
        elif choice == '3':
            demo_tsp_explanation()
        elif choice == '4':
            compare_bco_parameters()
        elif choice == '5':
            create_custom_tsp()
        elif choice == '0':
            print("👋 Tạm biệt! Cảm ơn bạn đã sử dụng BCO demo!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")


def create_custom_tsp():
    """
    Tạo TSP problem tùy chỉnh
    """
    print("\n🏗️ TẠO TSP PROBLEM TÙY CHỈNH")
    print("="*40)
    
    try:
        n_cities = int(input("Nhập số lượng thành phố (3-10): "))
        if n_cities < 3 or n_cities > 10:
            print("❌ Số thành phố phải từ 3-10!")
            return
        
        print(f"\n📍 Nhập tọa độ cho {n_cities} thành phố:")
        cities = []
        city_names = []
        
        for i in range(n_cities):
            name = input(f"Tên thành phố {i+1}: ").strip() or f"City_{i+1}"
            
            while True:
                try:
                    x = float(input(f"Tọa độ X của {name}: "))
                    y = float(input(f"Tọa độ Y của {name}: "))
                    break
                except ValueError:
                    print("❌ Vui lòng nhập số!")
            
            cities.append((x, y))
            city_names.append(name)
        
        # Tạo TSP problem
        tsp = TSProblem(cities, city_names)
        
        # Chạy BCO
        bco_tsp = BCO_TSP(
            tsp_problem=tsp,
            population_size=15,
            max_iterations=50,
            limit=5
        )
        
        print(f"\n🚀 Chạy BCO cho TSP {n_cities} thành phố...")
        best_tour, best_distance = bco_tsp.optimize_tsp(verbose=True)
        
        # Hiển thị kết quả
        tour_names = [city_names[i] for i in best_tour]
        print(f"\n🏆 TOUR TỐI ƯU:")
        print(f"Route: {' → '.join(tour_names)}")
        print(f"Distance: {best_distance:.2f}")
        
        # Visualization
        tsp.visualize_tour(best_tour, f"Custom TSP - {n_cities} Cities")
        bco_tsp.plot_convergence()
        
    except ValueError:
        print("❌ Input không hợp lệ!")
    except KeyboardInterrupt:
        print("\n👋 Demo bị hủy bởi người dùng!")


def main():
    """
    Main function để chạy demo
    """
    print("🐝" * 50)
    print("WELCOME TO SIMPLE BCO DEMO")
    print("Bee Colony Optimization - Đơn giản và dễ hiểu")
    print("🐝" * 50)
    
    print("\n📚 GIỚI THIỆU:")
    print("Đây là demo BCO đơn giản để hiểu cách thuật toán hoạt động.")
    print("Bạn sẽ thấy BCO giải quyết 2 loại bài toán:")
    print("1. Function Optimization (tìm minimum của hàm số)")
    print("2. TSP - Traveling Salesman Problem (tìm đường đi ngắn nhất)")
    
    print("\n🎯 MỤC TIÊU DEMO:")
    print("• Hiểu rõ từng bước của thuật toán BCO")
    print("• Thấy cách BCO hoạt động trên bài toán thực tế")
    print("• Học cách điều chỉnh parameters")
    print("• Trực quan hóa kết quả optimization")
    
    input("\n⏸️  Nhấn Enter để bắt đầu...")
    
    # Chạy interactive demo
    interactive_demo()


if __name__ == "__main__":
    main()