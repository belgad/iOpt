import numpy as np

from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.problem import Problem
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import Point


class _NestedProblem(Problem):
    """
    Class for nested one-dimensional problem
    """
    def __init__(self, curr_sub_dim: int, left_bound: float, right_bound: float):
        super(_NestedProblem, self).__init__()

        self.current_sub_dimension = curr_sub_dim

        self.name = "Nested problem"
        self.dimension = 1
        self.number_of_float_variables = 1
        self.float_variable_names = np.array(self.current_sub_dimension, ndmin=1, dtype=str)
        self.lower_bound_of_float_variables = np.array(left_bound, ndmin=1, dtype=np.double)
        self.upper_bound_of_float_variables = np.array(right_bound, ndmin=1, dtype=np.double)


class NestedGSATask(SearchData):
    """
    Special task class for nested dimensionality reduction scheme
    """
    def __init__(self, parameters: SolverParameters,
            curr_sub_dim: int,
            left_bound: float, right_bound: float,
            max_iter_count: int):
        super().__init__(_NestedProblem(curr_sub_dim, left_bound, right_bound))

        self.curr_iter_count: int = 0
        self.parameters: SolverParameters = parameters

        self.left_bound: float = left_bound
        self.right_bound: float = right_bound
        self.best: SearchDataItem = None
        self.new_x = self.left_bound

        self.recalc_r: bool = True
        self.m: float = 1.0
        self.min_delta: float = self.right_bound - self.left_bound
        # self.nested_task_iteration_count = int(
        #     pow(3 * self.parameters.global_method_iteration_count,
        #         1.0 / self.dimension)) + 1
        self.nested_task_iteration_count: int = max_iter_count

    def reset(self):
        """
        Reset search data of task
        """
        self._allTrials.clear()
        self._RGlobalQueue.Clear()
        self._firstDataItem = None

        self.curr_iter_count = 0
        self.best = None
        self.new_x = self.left_bound
        self.recalc_r = True
        self.m = 1.0
        self.min_delta = self.right_bound - self.left_bound

    def insert_data_item(self,
            func_value: np.double) -> None:
        """
        Insert function value for previously calculated `new_x` point
        """
        if self.curr_iter_count == 0:  # first point - left bound
            self._firstDataItem = SearchDataItem(
                Point([self.left_bound], []),
                np.double(self.left_bound)
            )
            self._firstDataItem.set_z(func_value)

            self._allTrials.append(self._firstDataItem)

            self.curr_iter_count = 1
        elif self.curr_iter_count == 1:  # second point - right bound
            new_data_item = SearchDataItem(
                Point([self.right_bound], []),
                np.double(self.right_bound)
            )
            new_data_item.set_z(func_value)

            self._firstDataItem.set_right(new_data_item)
            new_data_item.set_left(self._firstDataItem)

            self.calculate_m(new_data_item)

            self._allTrials.append(new_data_item)
            self._RGlobalQueue.insert(new_data_item.globalR, new_data_item)

            self.curr_iter_count = 2
        else:  # next points
            new_data_item = SearchDataItem(
                Point([self.new_x], []),
                np.double(self.new_x)
            )
            new_data_item.set_z(func_value)
            right_data_item = self.best
            left_data_item = right_data_item.get_left()

            left_data_item.set_right(new_data_item)
            new_data_item.set_left(left_data_item)
            new_data_item.set_right(right_data_item)
            right_data_item.set_left(new_data_item)

            self.calculate_r(new_data_item)
            self.calculate_m(new_data_item)
            self.calculate_r(right_data_item)
            self.calculate_m(right_data_item)

            self._allTrials.append(new_data_item)
            self._RGlobalQueue.insert(new_data_item.globalR, new_data_item)
            self._RGlobalQueue.insert(right_data_item.globalR, right_data_item)

            self.curr_iter_count += 1

    def calculate_iteration_point(self) -> float:
        """
        Calculate the point of a new trial
        """
        # first point - left bound
        if self.curr_iter_count == 0:
            return self.left_bound
        # second point - right bound
        elif self.curr_iter_count == 1:
            return self.right_bound
        else:
            # recalculate all intervals' characteristic
            if self.recalc_r:
                for item in self:
                    self.calculate_r(item)
                self.refill_queue()
                self.recalc_r = False
            self.best = self.get_data_item_with_max_global_r()

            xl = self.best.get_left().get_x()
            xr = self.best.get_x()
            zl = self.best.get_left().get_z()
            zr = self.best.get_z()
            self.new_x = float(0.5 * (xl + xr) - 0.5 * (zr - zl) / self.m / self.parameters.r)

            # update min_delta
            self.min_delta = min(self.min_delta, float(xr - xl))

            return self.new_x

    def calculate_r(self, curr_point: SearchDataItem) -> None:
        """
        Calculate the characteristic of an interval
        """
        if curr_point is None:
            raise RuntimeError("calculate_global_r: curr_point is None")
        left_point = curr_point.get_left()
        if left_point is None:
            curr_point.globalR = np.double(-np.inf)
            return
        zl = left_point.get_z()
        zr = curr_point.get_z()
        r = self.parameters.r
        deltax = curr_point.get_x() - left_point.get_x()
        curr_point.globalR = deltax\
            + (zr - zl) * (zr - zl) / (deltax * self.m * self.m * r * r)\
            - 2 * (zr + zl) / (self.m * r)

    def calculate_m(self, curr_point: SearchDataItem) -> None:
        """
        Calculate an estimate of Lipschitz constant
        """
        if curr_point is None:
            raise RuntimeError("calculate_global_r: curr_point is None")
        left_point = curr_point.get_left()
        if left_point is None:
            return
        m = float(abs(curr_point.get_z() - left_point.get_z())\
            / (curr_point.get_x() - left_point.get_x()))
        if m > self.m:
            self.m = m
            self.recalc_r = True

    def check_stop_condition(self) -> bool:
        """
        Check the stop condition using modified iterations count
        """
        return self.curr_iter_count >= self.nested_task_iteration_count or \
            bool(self.min_delta < self.parameters.eps)
