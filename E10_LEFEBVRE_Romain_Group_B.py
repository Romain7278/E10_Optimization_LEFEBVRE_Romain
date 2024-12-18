import numpy as np
from scipy.optimize import linprog

def branch_and_bound(revenues, days, max_days):
    """
    Solves a binary integer linear programming problem using the Branch and Bound method.

    Parameters:
    revenues (list): List of revenue values for each project.
    days (list): List of days required for each project.
    max_days (int): Maximum available days for allocation.

    Returns:
    tuple: A tuple containing the optimal solution (list of binary values) and the maximum revenue.
    """
    revenues = np.array(revenues)
    days = np.array(days)
    best_solution = None
    best_revenue = -np.inf

    def solve_relaxed_lp(fixed):
        """
        Solves the relaxed linear programming problem with fixed variables.

        Parameters:
        fixed (list): List of tuples representing fixed variable indices and their values.

        Returns:
        scipy.optimize.OptimizeResult: Result of the relaxed LP solver.
        """
        n = len(revenues)
        bounds = [(0, 1) for _ in range(n)]
        for var, value in fixed:
            bounds[var] = (value, value)

        res = linprog(-revenues, A_ub=[days], b_ub=[max_days], bounds=bounds, method='highs')
        return res

    def branch(current_solution, fixed):
        """
        Performs the branching process to explore solutions in the search tree.

        Parameters:
        current_solution (list): Current solution from the relaxed LP.
        fixed (list): List of tuples representing fixed variable indices and their values.
        """
        nonlocal best_solution, best_revenue

        if all((xi == 0 or xi == 1) for xi in current_solution):
            revenue = sum(revenues[i] for i, xi in enumerate(current_solution) if xi == 1)
            if revenue > best_revenue:
                best_revenue = revenue
                best_solution = current_solution
            return

        for i, xi in enumerate(current_solution):
            if not (xi == 0 or xi == 1):
                break

        for branch_value in [0, 1]:
            new_fixed = fixed + [(i, branch_value)]
            res = solve_relaxed_lp(new_fixed)

            if res.success:
                branch(res.x, new_fixed)

    res = solve_relaxed_lp([])

    if not res.success:
        print("No feasible solution found in the problem.")
        return None, None

    branch(res.x, [])
    return best_solution, best_revenue

if __name__ == "__main__":
    projects = [1, 2, 3, 4, 5, 6]
    revenues = [15, 20, 5, 25, 22, 17]
    days = [51, 60, 35, 60, 53, 10]
    max_days = 100

    solution, revenue = branch_and_bound(revenues, days, max_days)

    if solution is not None:
        print("Optimal strategy:")
        for i, x in enumerate(solution):
            print(f"Project {projects[i]}: {'Selected' if x == 1 else 'Not Selected'}")
        print(f"Maximum Revenue: {revenue}")
    else:
        print("No feasible solution found.")