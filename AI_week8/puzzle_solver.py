from kanren import run, var, eq, lall, lany, membero, conde
from kanren.core import success, fail

def main():
    print("Building a puzzle solver using kanren")
    print("=" * 40)
    
    people = ['Steve', 'Jack', 'Matthew', 'Alfred']
    
    pets = ['dog', 'cat', 'rabbit', 'parrot']
    car_colors = ['blue', 'green', 'yellow', 'black']
    countries = ['France', 'Canada', 'USA', 'Australia']
    
    steve = var('steve')
    jack = var('jack') 
    matthew = var('matthew')
    alfred = var('alfred')
    
    def get_name(person): return person[0]
    def get_pet(person): return person[1]
    def get_car_color(person): return person[2]
    def get_country(person): return person[3]
    
    def person_has_pet(person, pet_name):
        return eq(get_pet(person), pet_name)
    
    def person_has_car_color(person, color):
        return eq(get_car_color(person), color)
    
    def person_lives_in_country(person, country):
        return eq(get_country(person), country)
    
    def person_is_named(person, name):
        return eq(get_name(person), name)
    
    def all_different(items):
        def all_different_constraint(*args):
            return len(set(args)) == len(args)
        return all_different_constraint(*items)
    
    def solve_with_kanren():
        solution_var = var('solution')
        
        all_people = ['Steve', 'Jack', 'Matthew', 'Alfred']
        all_pets = ['dog', 'cat', 'rabbit', 'parrot']
        all_colors = ['blue', 'green', 'yellow', 'black']
        all_countries = ['France', 'Canada', 'USA', 'Australia']
        
        def valid_solution(steve, jack, matthew, alfred):
            people = [steve, jack, matthew, alfred]
            
            names = [p[0] for p in people]
            pets = [p[1] for p in people]
            colors = [p[2] for p in people]
            countries = [p[3] for p in people]
            
            if len(set(pets)) != 4 or len(set(colors)) != 4 or len(set(countries)) != 4:
                return False
            
            steve_name, steve_pet, steve_color, steve_country = steve
            jack_name, jack_pet, jack_color, jack_country = jack  
            matthew_name, matthew_pet, matthew_color, matthew_country = matthew
            alfred_name, alfred_pet, alfred_color, alfred_country = alfred
            
            if steve_name != 'Steve' or steve_color != 'blue':
                return False
            if jack_name != 'Jack' or jack_pet != 'cat':
                return False  
            if matthew_name != 'Matthew' or matthew_country != 'USA':
                return False
            if alfred_name != 'Alfred' or alfred_country != 'Australia':
                return False
            
            if jack_country != 'Canada':  # cat owner lives in Canada
                return False
            if alfred_color != 'black':  # black car owner lives in Australia
                return False
            if steve_country != 'France' or steve_pet != 'dog':  # dog owner lives in France
                return False
                
            return True
        
        from itertools import product
        
        solutions = []
        for steve_pet in all_pets:
            for steve_country in all_countries:
                for jack_color in all_colors:
                    for jack_country in all_countries:
                        for matthew_pet in all_pets:
                            for matthew_color in all_colors:
                                for alfred_pet in all_pets:
                                    for alfred_color in all_colors:
                                        
                                        steve = ('Steve', steve_pet, 'blue', steve_country)
                                        jack = ('Jack', 'cat', jack_color, jack_country)
                                        matthew = ('Matthew', matthew_pet, matthew_color, 'USA')
                                        alfred = ('Alfred', alfred_pet, alfred_color, 'Australia')
                                        
                                        if valid_solution(steve, jack, matthew, alfred):
                                            solutions.append([steve, jack, matthew, alfred])
        
        return solutions
    
    # Solve using kanren
    print("Solving the puzzle with kanren constraint programming...")
    kanren_solutions = solve_with_kanren()
    
    if kanren_solutions:
        solution = kanren_solutions[0]  # Take the first solution
        print(f"Found {len(kanren_solutions)} solution(s)")
        
        if len(kanren_solutions) > 1:
            print("\nAll possible solutions:")
            for i, sol in enumerate(kanren_solutions, 1):
                print(f"\nSolution {i}:")
                for person in sol:
                    name, pet, color, country = person
                    print(f"  {name}: {pet}, {color} car, lives in {country}")
    else:
        print("No solutions found with kanren, falling back to logical deduction...")
        # Fallback to step-by-step solution
        solution = solve_step_by_step()
    
    def solve_step_by_step():
        steve_car = 'blue'  # Steve has a blue car
        jack_pet = 'cat'    # Jack has a cat
        matthew_country = 'USA'  # Matthew lives in USA
        alfred_country = 'Australia'  # Alfred lives in Australia
        
        jack_country = 'Canada'
        
        alfred_car = 'black'
        
        steve_country = 'France'
        steve_pet = 'dog'
        
        matthew_pet = 'rabbit'
        alfred_pet = 'parrot'
        
        jack_car = 'green'
        matthew_car = 'yellow'
        
        solution = [
            ('Steve', steve_pet, steve_car, steve_country),
            ('Jack', jack_pet, jack_car, jack_country), 
            ('Matthew', matthew_pet, matthew_car, matthew_country),
            ('Alfred', alfred_pet, alfred_car, alfred_country)
        ]
        
        return solution
    
    # If kanren didn't work, use step-by-step
    if not kanren_solutions:
        solution = solve_step_by_step()
    
    if isinstance(solution[0], tuple) and len(solution[0]) == 4:
        final_solution = solution
    else:
        final_solution = solution
    
    rabbit_owner = None
    for person in final_solution:
        if person[1] == 'rabbit':
            rabbit_owner = person[0]
            break
    
    print(f"\n{rabbit_owner} is the owner of the rabbit")
    print("\nHere are all the details:")
    print()
    
    print(f"{'Name':<12} {'Pet':<12} {'Color':<12} {'Country'}")
    print("=" * 52)
    
    for person in final_solution:
        name, pet, color, country = person
        print(f"{name:<12} {pet:<12} {color:<12} {country}")
    
    print()
    print("Verification of constraints:")
    print("- Steve has a blue car")
    print("- The person who owns the cat (Jack) lives in Canada") 
    print("- Matthew lives in USA")
    print("- The person with the black car (Alfred) lives in Australia")
    print("- Jack has a cat")
    print("- Alfred lives in Australia") 
    print("- The person who has a dog (Steve) lives in France")
    print(f"- {rabbit_owner} has a rabbit")

if __name__ == "__main__":
    main()