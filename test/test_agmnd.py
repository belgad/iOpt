import unittest

import numpy as np

from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.problem import Problem
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import FunctionValue, Point
from problems.rastrigin import Rastrigin
from problems.xsquared import XSquared


CONSOLE_OUTPUT_MODE = 'full'


class TestSolveXSquared(unittest.TestCase):
    def setUp(self):
        self.problem = XSquared(1)
        params = SolverParameters(r=3.5, eps=1e-6)
        self.solver = Solver(self.problem, parameters=params)
        self.cfol = ConsoleOutputListener(mode=CONSOLE_OUTPUT_MODE)
        self.solver.add_listener(self.cfol)

    def test_solve(self):
        print("XSquared")
        print("Default Method solve")
        sol = self.solver.solve()
        self.assertAlmostEqual(
            sol.best_trials[0].point.float_variables[0],
            self.problem.known_optimum[0].point.float_variables[0],
            delta=1e-2)
        print("NumericalDerivativesMethod solve")
        sol = self.solver.agmnd_solve()
        self.assertAlmostEqual(
            sol.best_trials[0].point.float_variables[0],
            self.problem.known_optimum[0].point.float_variables[0],
            delta=1e-2)


class TestSolveRastrigin(unittest.TestCase):
    def setUp(self):
        self.problem = Rastrigin(1)
        params = SolverParameters(r=3.5, eps=1e-6)
        self.solver = Solver(self.problem, parameters=params)
        self.cfol = ConsoleOutputListener(mode=CONSOLE_OUTPUT_MODE)
        self.solver.add_listener(self.cfol)

    def test_solve(self):
        print("Rastrigin")
        print("Default Method solve")
        sol = self.solver.solve()
        self.assertAlmostEqual(
            sol.best_trials[0].point.float_variables[0],
            self.problem.known_optimum[0].point.float_variables[0],
            delta=1e-5)
        print("NumericalDerivativesMethod solve")
        sol = self.solver.agmnd_solve()
        self.assertAlmostEqual(
            sol.best_trials[0].point.float_variables[0],
            self.problem.known_optimum[0].point.float_variables[0],
            delta=1e-5)


class SinusoidProblem(Problem):
    def __init__(self):
        super(SinusoidProblem, self).__init__()
        self.name = "Sinusoid"
        self.dimension = 1
        self.number_of_float_variables = 1
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.float_variable_names = np.array(["x"], dtype=str)
        self.lower_bound_of_float_variables = np.array([0], dtype=np.double)
        self.upper_bound_of_float_variables = np.array([8], dtype=np.double)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        function_value.value = np.sin(point.float_variables[0])
        return function_value


class TestSolveSinusoid(unittest.TestCase):
    def setUp(self):
        self.problem = SinusoidProblem()
        params = SolverParameters(r=3.5, eps=1e-6)
        self.solver = Solver(self.problem, parameters=params)
        self.cfol = ConsoleOutputListener(mode=CONSOLE_OUTPUT_MODE)
        self.solver.add_listener(self.cfol)

    def test_solve(self):
        print("Sinusoid")
        print("Default Method solve")
        sol = self.solver.solve()
        self.assertAlmostEqual(
            sol.best_trials[0].point.float_variables[0],
            np.pi * 3 / 2,
            delta=1e-5)
        print("NumericalDerivativesMethod solve")
        sol = self.solver.agmnd_solve()
        self.assertAlmostEqual(
            sol.best_trials[0].point.float_variables[0],
            np.pi * 3 / 2,
            delta=1e-5)


if __name__ == '__main__':
    unittest.main()
