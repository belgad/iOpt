from enum import Enum

from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from problems.GKLS import GKLS
from problems.grishagin import Grishagin
from problems.hill import Hill
from problems.shekel import Shekel


class SolverType(Enum):
    DEFAULT = 1
    AGMND = 2


class ProblemType(Enum):
    HILL = 1
    SHEKEL = 2
    GRISHAGIN = 3
    GKLS = 4


R = 3.5
EPS = 1e-2
ITERS_LIMIT = 100_000
SOLVER = SolverType.DEFAULT
PROBLEM_TYPE = ProblemType.GKLS

if __name__ == '__main__':
    if SOLVER == SolverType.DEFAULT:
        print('-------------------- DEFAULT SOLVER')
    elif SOLVER == SolverType.AGMND:
        print('-------------------- AGMND SOLVER')
    else:
        raise ValueError("Wrong solver type!")

    if PROBLEM_TYPE in (ProblemType.HILL, ProblemType.SHEKEL):
        problem_count = 1000
    elif PROBLEM_TYPE in (ProblemType.GRISHAGIN, ProblemType.GKLS):
        problem_count = 100
    else:
        raise ValueError("Wrong problem type!")

    for i in range(problem_count):
        if PROBLEM_TYPE == ProblemType.HILL:
            problem = Hill(i)
        elif PROBLEM_TYPE == ProblemType.SHEKEL:
            problem = Shekel(i)
        elif PROBLEM_TYPE == ProblemType.GRISHAGIN:
            problem = Grishagin(i)
        elif PROBLEM_TYPE == ProblemType.GKLS:
            problem = GKLS(3, i + 1)
        params = SolverParameters(r=R, eps=EPS, iters_limit=ITERS_LIMIT)
        solver = Solver(problem, params)

        if SOLVER == SolverType.DEFAULT:
            solution = solver.solve()
        elif SOLVER == SolverType.AGMND:
            solution = solver.agmnd_solve()

        print(
            'iters:', solution.number_of_global_trials,
            '|',
            'time:', solution.solving_time,
            sep='', end='\n'
        )
