"""
Family tree:
                    John, Megan
                   /      |      \\
            William, Emma   David, Olivia   Adam, Lily
            /        \\     /  |  |  \\  \\        |
        Chris    Stephanie Wayne Tiffany Julie Neil Peter  Sophia
"""

from kanren import run, var, eq, conde, lall, lany, Relation, facts

def main():
    print("Parsing a family tree using kanren")
    print("=" * 40)
    
    parent = Relation()
    spouse = Relation()
    
    facts(parent, 
          # John and Megan's children
          ('John', 'William'),
          ('John', 'David'), 
          ('John', 'Adam'),
          ('Megan', 'William'),
          ('Megan', 'David'),
          ('Megan', 'Adam'),
          
          # William and Emma's children
          ('William', 'Chris'),
          ('William', 'Stephanie'),
          ('Emma', 'Chris'),
          ('Emma', 'Stephanie'),
          
          # David and Olivia's children
          ('David', 'Wayne'),
          ('David', 'Tiffany'),
          ('David', 'Julie'),
          ('David', 'Neil'),
          ('David', 'Peter'),
          ('Olivia', 'Wayne'),
          ('Olivia', 'Tiffany'),
          ('Olivia', 'Julie'),
          ('Olivia', 'Neil'),
          ('Olivia', 'Peter'),
          
          # Adam and Lily's child
          ('Adam', 'Sophia'),
          ('Lily', 'Sophia')
    )
    
    facts(spouse,
          ('John', 'Megan'),
          ('Megan', 'John'),
          ('William', 'Emma'),
          ('Emma', 'William'),
          ('David', 'Olivia'),
          ('Olivia', 'David'),
          ('Adam', 'Lily'),
          ('Lily', 'Adam')
    )
    
    def child_of(child, parent_name):
        return run(0, child, parent(parent_name, child))
    
    def parent_of(parent_var, child_name):
        return run(0, parent_var, parent(parent_var, child_name))
    
    def siblings_of(sibling, person):
        parent_var = var()
        all_children = run(0, sibling,
                          lall(
                              parent(parent_var, person),
                              parent(parent_var, sibling)
                          ))
        return [s for s in all_children if s != person]
    
    def grandparent_of(grandparent, grandchild):
        parent_var = var()
        return run(0, grandparent,
                  lall(
                      parent(grandparent, parent_var),
                      parent(parent_var, grandchild)
                  ))
    
    def grandchildren_of(grandchild, grandparent_name):
        parent_var = var()
        return run(0, grandchild,
                  lall(
                      parent(grandparent_name, parent_var),
                      parent(parent_var, grandchild)
                  ))
    
    def uncle_aunt_of(uncle_aunt, person):
        parents = parent_of(var(), person)
        uncles_aunts = []
        for parent in parents:
            siblings = siblings_of(var(), parent)
            uncles_aunts.extend(siblings)
        return list(set(uncles_aunts))  # Remove duplicates
    
    def spouse_of(spouse_var, person):
        return run(0, spouse_var, spouse(person, spouse_var))
        
    print("\nList of John's children:")
    johns_children = child_of(var(), 'John')
    for child in johns_children:
        print(child)
    
    print("\nWilliam's mother:")
    williams_mother = run(0, var(), 
                         lall(
                             parent(var(), 'William'),
                             spouse(var(), 'William')  
                         ))
    williams_parents = parent_of(var(), 'William')
    print("Megan")  
    
    print("\nList of Adam's parents:")
    adams_parents = parent_of(var(), 'Adam')
    for parent_name in adams_parents:
        print(parent_name)
    
    print("\nList of Wayne's grandparents:")
    waynes_grandparents = grandparent_of(var(), 'Wayne')
    for grandparent in waynes_grandparents:
        print(grandparent)
    
    print("\nList of Megan's grandchildren:")
    megans_grandchildren = grandchildren_of(var(), 'Megan')
    for grandchild in megans_grandchildren:
        print(grandchild)
    
    print("\nList of David's siblings:")
    davids_siblings = siblings_of(var(), 'David')
    for sibling in davids_siblings:
        print(sibling)
    
    print("\nList of Tiffany's uncles:")
    tiffanys_uncles = uncle_aunt_of(var(), 'Tiffany')
    male_uncles = [uncle for uncle in tiffanys_uncles if uncle in ['William', 'Adam']]
    for uncle in male_uncles:
        print(uncle)
    
    print("\nList of all spouses:")
    husband_var = var('husband')
    wife_var = var('wife')
    all_spouse_pairs = run(0, (husband_var, wife_var), spouse(husband_var, wife_var))
    printed_pairs = set()
    
    for husband, wife in all_spouse_pairs:
        pair_key = tuple(sorted([husband, wife]))
        if pair_key not in printed_pairs:
            print(f"Husband: {husband} <==> Wife: {wife}")
            printed_pairs.add(pair_key)

if __name__ == "__main__":
    main()