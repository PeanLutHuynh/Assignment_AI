from kanren import run, var, eq, conde, lall, lany

def main():
    print("Matching mathematical expressions using kanren")
    print("=" * 50)
    
    # Example 1: Simple pattern matching
    print("\n1. Simple pattern matching:")
    print("Pattern: ('*', ?x, 3) should match ('*', 2, 3)")
    
    x = var('x')
    
    pattern = ('*', x, 3)
    expression = ('*', 2, 3)
    
    solutions = run(0, x, eq(pattern, expression))
    print(f"Solutions for x: {solutions}")
    
    # Example 2: Multiple variables
    print("\n2. Multiple variables pattern:")
    print("Pattern: ('+', ?a, ?b) should match ('+', 4, 7)")
    
    a, b = var('a'), var('b')
    pattern2 = ('+', a, b)
    expression2 = ('+', 4, 7)
    
    solutions2 = run(0, (a, b), eq(pattern2, expression2))
    print(f"Solutions for (a, b): {solutions2}")
    
    # Example 3: Nested expressions
    print("\n3. Nested expression matching:")
    print("Pattern: ('+', ('*', ?x, ?y), ?z) should match ('+', ('*', 2, 3), 5)")
    
    x, y, z = var('x'), var('y'), var('z')
    pattern3 = ('+', ('*', x, y), z)
    expression3 = ('+', ('*', 2, 3), 5)
    
    solutions3 = run(0, (x, y, z), eq(pattern3, expression3))
    print(f"Solutions for (x, y, z): {solutions3}")
    
    # Example 4: Commutative matching
    print("\n4. Commutative properties:")
    print("Finding if expressions match under commutativity")
    
    def commutative_add(expr1, expr2):
        return lany(
            eq(expr1, expr2),  
            (lambda expr1, expr2: 
             eq(expr1, ('+', expr2[2], expr2[1])) if 
             isinstance(expr2, tuple) and len(expr2) == 3 and expr2[0] == '+' 
             else False)(expr1, expr2)
        )
    
    a, b = var('a'), var('b')
    expr_a = ('+', a, b)
    expr_b = ('+', 3, 2)
    expr_c = ('+', 2, 3)  
    
    comm_solutions = run(0, (a, b), 
                        lany(
                            eq(expr_a, expr_b),
                            eq(expr_a, expr_c)
                        ))
    print(f"Commutative solutions for (a, b): {comm_solutions}")
    
    # Example 5: System of equations
    print("\n5. System of equations:")
    print("Pattern 1: ('*', ?a, ?b) matches ('*', 2, 3)")
    print("Pattern 2: ('+', ?a, ?c) matches ('+', 2, 5)")
    print("Find common value for ?a")
    
    a, b, c = var('a'), var('b'), var('c')
    
    solutions_system = run(0, (a, b, c), 
                          lall(
                              eq(('*', a, b), ('*', 2, 3)),
                              eq(('+', a, c), ('+', 2, 5))
                          ))
    print(f"System solutions (a, b, c): {solutions_system}")
    
    # Example 6: More complex pattern matching
    print("\n6. Complex pattern matching:")
    print("Find all ways to match pattern (?op, ?x, ?y) with various expressions")
    
    op, x, y = var('op'), var('x'), var('y')
    pattern_complex = (op, x, y)
    
    expressions = [
        ('*', 2, 3),
        ('+', 4, 5),
        ('-', 10, 7),
        ('/', 15, 3)
    ]
    
    for expr in expressions:
        result = run(0, (op, x, y), eq(pattern_complex, expr))
        print(f"  Pattern matches {expr}: {result}")
    
    # Example 7: Associative properties with kanren
    print("\n7. Associative properties:")
    print("Testing (a + b) + c = a + (b + c)")
    
    def associative_test():
        a, b, c = var('a'), var('b'), var('c')
        
        left_assoc = (('+', ('+', a, b), c))  
        right_assoc = (('+', a, ('+', b, c)))  
        
        test_values = run(0, (a, b, c), 
                         lall(
                             eq(a, 1),
                             eq(b, 2), 
                             eq(c, 3)
                         ))
        
        if test_values:
            a_val, b_val, c_val = test_values[0]
            print(f"  With a={a_val}, b={b_val}, c={c_val}:")
            print(f"  Left form: (({a_val} + {b_val}) + {c_val}) = {(a_val + b_val) + c_val}")
            print(f"  Right form: ({a_val} + ({b_val} + {c_val})) = {a_val + (b_val + c_val)}")
            print(f"  Equal: {(a_val + b_val) + c_val == a_val + (b_val + c_val)}")
    
    associative_test()
    
    # Example 8: Advanced constraint solving
    print("\n8. Advanced constraint solving:")
    print("Find x, y such that x + y = 10 and x * y = 21")
    
    x, y = var('x'), var('y')
    
    # Kanren doesn't have built-in arithmetic
    # => Enumerate possible solutions
    possible_solutions = []
    for i in range(1, 10):
        for j in range(1, 10):
            if i + j == 10 and i * j == 21:
                possible_solutions.append((i, j))
    
    print(f"Mathematical solutions: {possible_solutions}")
    
    # Use kanren to verify these solutions
    for sol in possible_solutions:
        verification = run(0, (x, y), 
                          lall(
                              eq(x, sol[0]),
                              eq(y, sol[1])
                          ))
        if verification:
            print(f"  Kanren verified: x={sol[0]}, y={sol[1]}")

if __name__ == "__main__":
    main()