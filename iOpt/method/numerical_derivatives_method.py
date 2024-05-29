import copy

import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.method import Method
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import Point, FunctionValue


class NumericalDerivativesMethod(Method):
    """
    The NumericalDerivativesMethod Class contains an implementation of
    the One-Dimensional Global Optimization Algorithm utilizing Numerical Derivatives
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 calculator: Calculator = None):
        """
        NumericalDerivativesMethod class constructor

        :param parameters: parameters for solving the optimization problem.
        :param task: problem wrapper.
        :param evolvent: Peano-Hilbert evolvent mapping the segment [0,1] to the multidimensional region D.
        :param search_data: data structure for storing accumulated search information.
        :param calculator: class containing trial methods (parallel and/or inductive circuit)
        """
        super(NumericalDerivativesMethod, self).__init__(
            parameters, task, evolvent, search_data, calculator)

    def first_iteration(self) -> list[SearchDataItem]:
        """
        Perform the first iteration of the Global Search Algorithm
        """
        left = SearchDataItem(Point(self.evolvent.get_image(0.0), None), 0.,
                              function_values=[FunctionValue()] * self.numberOfAllFunctions)
        right = SearchDataItem(Point(self.evolvent.get_image(1.0), None), 1.0,
                               function_values=[FunctionValue()] * self.numberOfAllFunctions)

        items: list[SearchDataItem] = []

        if self.parameters.start_point:
            number_of_point: int = self.parameters.number_of_parallel_points - 1
            h: float = 1.0 / (number_of_point + 1)

            ystart_point = Point(copy.copy(self.parameters.start_point.float_variables), None)
            xstart_point = self.evolvent.get_inverse_image(self.parameters.start_point.float_variables)

            itemstart_point = SearchDataItem(ystart_point, xstart_point,
                                             function_values=[FunctionValue()] * self.numberOfAllFunctions)

            is_add_start_point: bool = False

            for i in range(number_of_point):
                x = h * (i + 1)
                y = Point(self.evolvent.get_image(x), None)
                item = SearchDataItem(y, x,
                                      function_values=[FunctionValue()] * self.numberOfAllFunctions)
                if x < xstart_point < h * (i + 2):
                    items.append(item)
                    items.append(itemstart_point)
                    is_add_start_point = True
                else:
                    items.append(item)

            if not is_add_start_point:
                items.append(itemstart_point)
        else:

            number_of_point: int = self.parameters.number_of_parallel_points
            h: float = 1.0 / (number_of_point + 1)

            for i in range(number_of_point):
                x = h * (i + 1)
                y = Point(self.evolvent.get_image(x), None)
                item = SearchDataItem(y, x,
                                      function_values=[FunctionValue()] * self.numberOfAllFunctions)
                items.append(item)

        self.calculate_functionals(left)
        self.calculate_functionals(right)

        self.calculator.calculate_functionals_for_items(items)

        for item in items:
            self.update_optimum(item)

        left.delta = 0
        self.calculate_global_r(left, None)

        items[0].delta = self.calculate_delta(left, items[0], self.dimension)
        self.calculate_global_r(items[0], left)
        for id_item, item in enumerate(items):
            if id_item > 0:
                items[id_item].delta = self.calculate_delta(items[id_item - 1], items[id_item], self.dimension)
                self.calculate_global_r(items[id_item], items[id_item - 1])
                self.calculate_m(items[id_item], items[id_item - 1])

        right.delta = self.calculate_delta(items[-1], right, self.dimension)
        self.calculate_global_r(right, items[-1])

        self.search_data.insert_first_data_item(left, right)
        for item in items:
            self.search_data.insert_data_item(item, right)

        self.recalcR = True
        self.recalcM = True

        self.iterations_count = len(items)
        self.search_data.solution.number_of_global_trials = len(items)

        return items

    def calculate_next_point_coordinate(self, point: SearchDataItem) -> float:
        """
        Compute the point of a new trial :math:`x^{k+1}` in a given interval :math:`[x_{t-1},x_t]`

        :param point: interval given by its right point :math:`x_t`.

        :return: the point of a new trial :math:`x^{k+1}` in this interval.
        """
        left_point = point.get_left()
        if left_point is None:
            print("calculate_next_point_coordinate: left_point is None")
            raise RuntimeError("calculate_next_point_coordinate: left_point is None")
        if left_point.get_left() is None:
            left_deriv_est = self.calculate_derivative_estimate(point, left_point)
        else:
            left_deriv_est = self.calculate_derivative_estimate(left_point, left_point.get_left())
        xr = point.get_x()
        xl = left_point.get_x()
        m = self.M[point.get_index()] * self.parameters.r
        x1, x2, x3 = self.calculate_auxiliary_points(point, left_point)
        if x1 <= x3 <= x2:
            return x3
        else:
            minorant1 = left_point.get_z()\
                + left_deriv_est * (xr - xl) - m * (x1 - xl) * (x1 - xl) / 2.
            minorant2 = (left_deriv_est - m * (x1 - xl)) * (x2 - x1)\
                + m * (x2 - x1) * (x2 - x1) / 2. + minorant1
            if minorant1 <= minorant2:
                return x1
            else:
                return x2

    def calculate_m(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        """
        Calculate an estimate of the Gelder constant between curr_point and left_point

        :param curr_point: right interval point.
        :param left_point: left interval point.
        """
        if curr_point is None:
            print("calculate_m: curr_point is None")
            raise RuntimeError("calculate_m: curr_point is None")
        if left_point is None:
            return
        index = curr_point.get_index()
        if left_point.get_index() == index and index >= 0:
            curr_deriv_est = self.calculate_derivative_estimate(curr_point, left_point)
            if left_point.get_left() is None:
                left_deriv_est = curr_deriv_est
            else:
                left_deriv_est = self.calculate_derivative_estimate(
                    left_point, left_point.get_left())
            m1 = abs(curr_deriv_est - left_deriv_est) \
                / curr_point.delta
            m2 = -2. * (curr_point.get_z() - left_point.get_z()\
                        - left_deriv_est * (curr_point.get_x() - left_point.get_x()))\
                 / (curr_point.delta * curr_point.delta)
            m3 = 2. * (curr_point.get_z() - left_point.get_z()\
                        - curr_deriv_est * (curr_point.get_x() - left_point.get_x()))\
                 / (curr_point.delta * curr_point.delta)
            m = max(m1, m2, m3)
            if m > self.M[index]:
                self.M[index] = m
                self.recalcR = True

    def calculate_global_r(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        """
        Calculate the global characteristic of an interval [left_point, curr_point]

        :param curr_point: right interval point.
        :param left_point: left interval point.
        """
        if curr_point is None:
            print("calculate_global_r: Curr point is NONE")
            raise RuntimeError("calculate_global_r: Curr point is NONE")
        if left_point is None:
            curr_point.globalR = -np.infty
            return None
        
        x1, x2, x3 = self.calculate_auxiliary_points(curr_point, left_point)
        if left_point.get_left() is None:
            left_deriv_est = self.calculate_derivative_estimate(curr_point, left_point)
        else:
            left_deriv_est = self.calculate_derivative_estimate(left_point, left_point.get_left())
        xl = left_point.get_x()
        xr = curr_point.get_x()
        m = self.M[curr_point.get_index()] * self.parameters.r
        a = left_deriv_est - m * (x1 - xl)
        b = left_point.get_z() + left_deriv_est * (xr - xl) - m * (x1 - xl) * (x1 - xl) / 2.
        if x1 <= x3 <= x2:
            global_r = a * (x3 - xl) + m * (x3 - x1) * (x3 - x1) / 2. + b
        else:
            global_r = min(b, a * (x2 - x1) + m * (x2 - x1) * (x2 - x1) / 2. + b)
        curr_point.globalR = -global_r

    def calculate_derivative_estimate(self,
            curr_point: SearchDataItem,
            left_point: SearchDataItem) -> np.double:
        """
        Calculate the numerical estimation of the first derivative in given point

        :param curr_point: point where to calculate derivative.
        :param left_point: left interval point.
        """
        if curr_point is None:
            print("calculate_derivative_estimate: curr_point is None")
            raise RuntimeError("calculate_derivative_estimate: curr_point is None")
        if left_point is None:
            left_point = curr_point
            curr_point = curr_point.get_right()
        return (curr_point.get_z() - left_point.get_z()) / (curr_point.get_x() - left_point.get_x())

    def calculate_auxiliary_points(self, curr_point: SearchDataItem, left_point: SearchDataItem):
        """
        Calculate auxiliary points for the interval characteristic calculations

        :param curr_point: right interval point.
        :param left_point: left interval point.
        """
        if curr_point is None:
            print("calculate_auxiliary_points: curr_point is None")
            raise RuntimeError("calculate_auxiliary_points: curr_point is None")
        if left_point is None:
            print("calculate_auxiliary_points: left_point is None")
            raise RuntimeError("calculate_auxiliary_points: left_point is None")
        curr_deriv_est = self.calculate_derivative_estimate(curr_point, left_point)
        if left_point.get_left() is None:
            left_deriv_est = curr_deriv_est
        else:
            left_deriv_est = self.calculate_derivative_estimate(left_point, left_point.get_left())
        deriv_est_delta = curr_deriv_est - left_deriv_est
        zl = left_point.get_z()
        zr = curr_point.get_z()
        xl = left_point.get_x()
        xr = curr_point.get_x()
        m = self.M[curr_point.get_index()] * self.parameters.r
        d = (xr - xl - deriv_est_delta / m) / 2.
        x1 = ((zl - left_deriv_est * xl)
              - (zr - curr_deriv_est * xr)
              + m * (xr * xr - xl * xl) / 2.
              - m * d * d) / (m * (xr - xl) + deriv_est_delta)
        x2 = ((zl - left_deriv_est * xl)
              - (zr - curr_deriv_est * xr)
              + m * (xr * xr - xl * xl) / 2.
              + m * d * d) / (m * (xr - xl) + deriv_est_delta)
        x3 = x1 + xr - xl - left_deriv_est / m
        return x1, x2, x3
