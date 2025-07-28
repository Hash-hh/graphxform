import numpy as np


class ParetoArchive:
    """
    A class to manage a Pareto front archive for multi-objective optimization.

    The archive stores non-dominated solutions. Objectives are assumed to be maximized.
    """

    def __init__(self):
        # Stores list of tuples: ([obj1, obj2, ...], solution_data)
        self.front = []

    def get_front(self):
        """Returns the current Pareto front."""
        return self.front

    def update(self, new_solutions):
        """
        Updates the archive with a list of new solutions.

        Each solution in new_solutions should be a tuple: ([obj1, obj2, ...], data)
        e.g., ([0.75, 0.9], "C1=CC=CC=C1")
        """
        for new_sol in new_solutions:
            new_objs, new_data = new_sol
            is_dominated_by_archive = False

            # Use a copy of the front for safe iteration while modifying
            for i in range(len(self.front) - 1, -1, -1):
                archive_objs, _ = self.front[i]

                if self._dominates(archive_objs, new_objs):
                    # New solution is dominated by an existing one, so discard it
                    is_dominated_by_archive = True
                    break
                elif self._dominates(new_objs, archive_objs):
                    # New solution dominates an existing one, so remove the old one
                    del self.front[i]

            if not is_dominated_by_archive:
                self.front.append(new_sol)

    def _dominates(self, objs1, objs2):
        """Checks if solution 1 dominates solution 2 (maximization)."""
        all_ge = np.all(np.array(objs1) >= np.array(objs2))
        any_gt = np.any(np.array(objs1) > np.array(objs2))
        return all_ge and any_gt

    def _calculate_crowding_distance(self):
        """
        Calculates the crowding distance for each solution in the front.
        Inspired by the NSGA-II algorithm.
        """
        if not self.front:
            return []

        num_solutions = len(self.front)
        if num_solutions <= 2:
            # Not enough points to calculate distance, treat all as equally important
            return [(sol, float('inf')) for sol in self.front]

        objectives = np.array([sol[0] for sol in self.front])
        num_objectives = objectives.shape[1]
        distances = np.zeros(num_solutions)

        for i in range(num_objectives):
            # Sort solutions by the current objective
            sorted_indices = np.argsort(objectives[:, i])

            # Assign infinite distance to boundary solutions
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # Get max and min objective values for normalization
            obj_min = objectives[sorted_indices[0], i]
            obj_max = objectives[sorted_indices[-1], i]

            if obj_max == obj_min:
                continue

            # Calculate distance for intermediate solutions
            for j in range(1, num_solutions - 1):
                idx = sorted_indices[j]
                prev_idx = sorted_indices[j - 1]
                next_idx = sorted_indices[j + 1]

                distances[idx] += (objectives[next_idx, i] - objectives[prev_idx, i]) / (obj_max - obj_min)

        return [(self.front[i], distances[i]) for i in range(num_solutions)]

    def select_for_training(self, num_to_select: int, strategy='crowding_distance'):
        """
        Selects a subset of solutions from the front for training.
        """
        if not self.front:
            return []

        if len(self.front) <= num_to_select:
            return [sol[1] for sol in self.front]  # Return just the data part

        if strategy == 'crowding_distance':
            solutions_with_distances = self._calculate_crowding_distance()
            # Sort by distance (descending)
            solutions_with_distances.sort(key=lambda x: x[1], reverse=True)
            # Return the data of the top N solutions
            return [sol[0][1] for sol in solutions_with_distances[:num_to_select]]
        elif strategy == 'random':
            indices = np.random.choice(len(self.front), size=num_to_select, replace=False)
            return [self.front[i][1] for i in indices]
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")