from trust_bench.core.problem import Problem


class Registry:
    """A catalog of Problems, looked up by id or tag."""

    def __init__(self):
        self._by_id: dict[str, Problem] = {}

    def register(self, problem: Problem) -> Problem:
        if not callable(problem.residual):
            raise TypeError(f"Problem {problem.id!r} has no callable residual")
        self._by_id[problem.id] = problem
        return problem

    def get(self, id: str) -> Problem:
        return self._by_id[id]

    def by_tag(self, tag: str) -> list[Problem]:
        return [p for p in self._by_id.values() if tag in p.tags]
