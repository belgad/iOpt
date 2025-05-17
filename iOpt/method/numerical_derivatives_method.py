from typing import Tuple

import numpy as np
import numpy.typing as npt

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.method import Method
from iOpt.method.nested_one_dimensional_method import NestedGSATask
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import Point, FunctionValue


class NumericalDerivativesMethod(Method):
    """
    The NumericalDerivativesMethod Class contains an implementation of
    the Global Optimization Algorithm utilizing Numerical Derivatives
    with nested dimensionality reduction scheme for multidimensional tasks
    """
    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 calculator: Calculator = None):
        super().__init__(parameters, task, evolvent, search_data, calculator)

        # add array of search data for dimensions
        self.nested_task_iteration_count = int(
            pow(self.parameters.global_method_iteration_count,
                1.0 / self.dimension)) + 1
        self.nested_tasks = [
            NestedGSATask(
                self.parameters, i,
                self.task.problem.lower_bound_of_float_variables[i],
                self.task.problem.upper_bound_of_float_variables[i],
                self.nested_task_iteration_count
            ) for i in range(self.dimension - 1)
        ]
        self.curr_x: npt.NDArray[np.double]\
            = np.copy(self.task.problem.lower_bound_of_float_variables).astype(np.double)
        self.curr_iter_count = 0
        self.curr_best: SearchDataItem = None

    def calculate_delta(self,
                        l_point: SearchDataItem, r_point: SearchDataItem,
                        _=None) -> np.double:
        return r_point.get_x() - l_point.get_x()

    def first_iteration(self) -> list[SearchDataItem]:
        self.curr_best = None
        self.curr_iter_count = 0
        left_x = self.task.problem.lower_bound_of_float_variables[-1]
        right_x = self.task.problem.upper_bound_of_float_variables[-1]
        self.curr_x[-1] = left_x
        left_coords = np.copy(self.curr_x)
        right_coords = np.copy(self.curr_x)
        right_coords[-1] = right_x
        left = SearchDataItem(
            Point(left_coords, []),
            left_x,
            [FunctionValue()] * self.numberOfAllFunctions
        )
        right = SearchDataItem(
            Point(right_coords, []),
            right_x,
            [FunctionValue()] * self.numberOfAllFunctions
        )

        items: list[SearchDataItem] = [left]

        number_of_point: int = self.parameters.number_of_parallel_points
        h: float = float(right_x - left_x) / (number_of_point + 1)
        for i in range(number_of_point):
            x = left_x + h * (i + 1)
            y_coord = np.copy(self.curr_x)
            y_coord[-1] = x
            y = Point(y_coord, [])
            item = SearchDataItem(y, x, [FunctionValue()] * self.numberOfAllFunctions)
            items.append(item)

        items.append(right)

        self.calculator.calculate_functionals_for_items(items)

        for item in items:
            self.update_optimum(item)

        for id_item in range(1, len(items)):
            items[id_item].set_left(items[id_item - 1])

        left.delta = np.double(0.0)
        self.calculate_global_r(left, None)

        # pylint: disable-next=C0200
        for id_item in range(1, len(items)):
            items[id_item].delta = self.calculate_delta(items[id_item - 1], items[id_item])
            self.calculate_m(items[id_item], items[id_item - 1])
        for id_item in range(1, len(items)):
            self.calculate_global_r(items[id_item], items[id_item - 1])

        self.search_data.insert_first_data_item(left, right)
        for item in items[1:-1]:
            self.search_data.insert_data_item(item, right)

        self.recalcM = False
        self.recalcR = False

        self.iterations_count += len(items)
        self.search_data.solution.number_of_global_trials += len(items)
        self.curr_iter_count += len(items)

        self.min_delta = h

        return items

    def check_stop_condition(self) -> bool:
        # check for iteration count
        # if self.iterations_count >= self.parameters.global_method_iteration_count:
        if self.stop:
            return True
        if self.iterations_count >= self.parameters.global_method_iteration_count:
            self.stop = True
            return True
        if self.dimension == 1:
            self.stop = bool(self.min_delta < self.parameters.eps)
        elif self.min_delta < self.parameters.eps or \
                self.curr_iter_count >= self.nested_task_iteration_count:
            self.update_nested_task(self.dimension - 2)
            if not self.stop:
                self.first_iteration()
        return self.stop

    def update_nested_task(self, dimension_id: int) -> None:
        """
        Push nested search data into selected dimension
        """
        if self.curr_best is None:
            raise RuntimeError("update_nested_task: self.curr_best is None")
        if dimension_id < 0:
            self.min_delta = np.sqrt(
                sum(map(np.square,[task.min_delta for task in self.nested_tasks])) +
                np.square(self.min_delta))
            self.stop = True
            return
        self.nested_tasks[dimension_id].insert_data_item(self.curr_best.get_z())
        if self.nested_tasks[dimension_id].check_stop_condition():
            self.update_nested_task(dimension_id - 1)
            self.nested_tasks[dimension_id].reset()
        self.curr_x[dimension_id] = self.nested_tasks[dimension_id].calculate_iteration_point()

    def calculate_next_point_coordinate(self, point: SearchDataItem) -> np.double:
        if point is None:
            raise RuntimeError("calculate_next_point_coordinate: point is None")
        left_point = point.get_left()
        if left_point is None:
            raise RuntimeError("calculate_next_point_coordinate: left_point is None")
        if left_point.get_left() is None:
            dl = self.calculate_derivative_estimate(point, left_point)
        else:
            dl = self.calculate_derivative_estimate(left_point, left_point.get_left())
        xr = point.get_x()
        xl = left_point.get_x()
        m = self.M[point.get_index()] * self.parameters.r
        x1, x2, x3 = self.calculate_auxiliary_points(point, left_point)
        if x1 <= x3 <= x2:
            return x3
        minorant1 = left_point.get_z() + dl * (xr - xl) - m * (x1 - xl) * (x1 - xl) / 2.0
        minorant2 = (dl - m * (x1 - xl)) * (x2 - x1) + m * (x2 - x1) * (x2 - x1) / 2.0 + minorant1
        return x1 if minorant1 <= minorant2 else x2

    def calculate_iteration_point(self) -> Tuple[SearchDataItem, SearchDataItem]:
        if self.recalcM:
            self.recalc_m()
        if self.recalcR:
            self.recalc_all_characteristics()
        old = self.search_data.get_data_item_with_max_global_r()
        self.min_delta = min(self.min_delta, old.delta)
        newx = self.calculate_next_point_coordinate(old)
        self.curr_x[-1] = newx
        newy = np.copy(self.curr_x)
        new = SearchDataItem(Point(newy, []), newx, [FunctionValue()] * self.numberOfAllFunctions)
        self.search_data.solution.number_of_global_trials += 1
        return new, old

    def calculate_m(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        if curr_point is None:
            raise RuntimeError("calculate_m: curr_point is None")
        if left_point is None:
            return
        index = curr_point.get_index()
        if left_point.get_index() != index or index < 0:
            return
        xr = curr_point.get_x()
        xl = left_point.get_x()
        zr = curr_point.get_z()
        zl = left_point.get_z()
        dr = self.calculate_derivative_estimate(curr_point, left_point)
        if left_point.get_left() is None:
            dl = dr
        else:
            dl = self.calculate_derivative_estimate(left_point, left_point.get_left())
        m1 = abs(dr - dl) / (xr - xl)
        m2 = -2 * (zr - zl - dl * (xr - xl)) / ((xr - xl) * (xr - xl))
        m3 = 2 * (zr - zl - dr * (xr - xl)) / ((xr - xl) * (xr - xl))
        m = max(m1, m2, m3)
        if m > self.M[index]:
            self.M[index] = m
            self.recalcR = True

    def calculate_global_r(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        if curr_point is None:
            raise RuntimeError("calculate_global_r: curr_point is None")
        if left_point is None:
            curr_point.globalR = -np.inf
            return
        x1, x2, x3 = self.calculate_auxiliary_points(curr_point, left_point)
        if left_point.get_left() is None:
            dl = self.calculate_derivative_estimate(curr_point, left_point)
        else:
            dl = self.calculate_derivative_estimate(left_point, left_point.get_left())
        xl = left_point.get_x()
        xr = curr_point.get_x()
        m = self.M[curr_point.get_index()] * self.parameters.r
        a = dl - m * (x1 - xl)
        b = left_point.get_z() + dl * (xr - xl) - m * (x1 - xl) * (x1 - xl) / 2.0
        if x1 <= x3 <= x2:
            global_r = a * (x3 - xl) + m * (x3 - x1) * (x3 - x1) / 2.0 + b
        else:
            global_r = min(b, a * (x2 - x1) + m * (x2 - x1) * (x2 - x1) / 2.0 + b)
        curr_point.globalR = -global_r

    def calculate_derivative_estimate(self,
            curr_point: SearchDataItem,
            left_point: SearchDataItem) -> np.double:
        """
        Calculate the numerical estimation of the first derivative in given point
        """
        if curr_point is None:
            raise RuntimeError("calculate_derivative_estimate: curr_point is None")
        return \
            ((curr_point.get_right().get_z() - curr_point.get_z()) / \
            (curr_point.get_right().get_x() - curr_point.get_x())) \
            if left_point is None else \
            ((curr_point.get_z() - left_point.get_z()) / \
            (curr_point.get_x() - left_point.get_x()))

    def calculate_auxiliary_points(self,
            curr_point: SearchDataItem,
            left_point: SearchDataItem) -> tuple[np.double, np.double, np.double]:
        """
        Calculate auxiliary points for the interval characteristic calculations
        """
        if curr_point is None:
            raise RuntimeError("calculate_auxiliary_points: curr_point is None")
        if left_point is None:
            raise RuntimeError("calculate_auxiliary_points: left_point is None")
        dr = self.calculate_derivative_estimate(curr_point, left_point)
        if left_point.get_left() is None:
            dl = dr
        else:
            dl = self.calculate_derivative_estimate(left_point, left_point.get_left())
        d_delta = dr - dl
        zl = left_point.get_z()
        zr = curr_point.get_z()
        xl = left_point.get_x()
        xr = curr_point.get_x()
        m = self.M[curr_point.get_index()] * self.parameters.r
        d = (xr - xl - d_delta / m) / 2.0
        x1 = (zl - dl * xl - zr + dr * xr + m * (xr * xr - xl * xl) / 2.0 - m * d * d) / \
            (m * (xr - xl) + d_delta)
        x2 = (zl - dl * xl - zr + dr * xr + m * (xr * xr - xl * xl) / 2.0 + m * d * d) / \
            (m * (xr - xl) + d_delta)
        x3 = x1 + xr - xl - dl / m
        return x1, x2, x3

    def update_optimum(self, point: SearchDataItem) -> None:
        super().update_optimum(point)
        if self.curr_best is None or point.get_z() < self.curr_best.get_z():
            self.curr_best = point

    def finalize_iteration(self) -> None:
        super().finalize_iteration()
        self.curr_iter_count += 1

    def renew_search_data(self, newpoint: SearchDataItem, oldpoint: SearchDataItem) -> None:
        newpoint.set_left(oldpoint.get_left())
        return super().renew_search_data(newpoint, oldpoint)
