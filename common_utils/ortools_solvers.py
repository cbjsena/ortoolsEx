import json
import logging
import time
from math import sqrt

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from core.decorators import log_solver_solve
from .base_solver import BaseSolver
from .common_run_opt import export_ortools_solver, export_cp_model

logger = logging.getLogger(__name__)


class BaseOrtoolsLinearSolver(BaseSolver):
    """
    OR-Tools의 pywraplp (LP, MIP) 솔버를 위한 기본 클래스.
    """

    def __init__(self, input_data, solver_name):
        super().__init__(input_data)
        self.solver = pywraplp.Solver.CreateSolver(solver_name)
        self.status_map = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        }
        if not self.solver:
            raise Exception(f"{solver_name} Solver not available.")

    @log_solver_solve
    def solve(self):
        try:
            self._create_variables()
            self._add_constraints()
            self._set_objective_function()

            status = self.solver.Solve()
            processing_time = self.get_time(self.solver.WallTime() / 1000.0)
            export_ortools_solver(self.solver, f'{self.problem_type}.mps')
            self.log_solve_resulte(self.status_map.get(status, "UNKNOWN"), processing_time)

            if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
                if status == pywraplp.Solver.FEASIBLE:
                    msg = "Feasible solution found, but it might not be optimal."
                    logger.warning(msg)

                results = self._extract_results()
                error_msg = None
                if status == pywraplp.Solver.FEASIBLE:
                    logger.warning(f"Feasible solution found for {self.problem_type}.")

                return results, error_msg, processing_time
            else:
                error_msg = f"Optimal solution not found. Solver status: {status}"
                return None, error_msg, processing_time

        except Exception as e:
            # BaseSolver의 에러 처리 로직을 그대로 활용
            return super().solve()


class BaseOrtoolsCpSolver(BaseSolver):
    """
    OR-Tools의 CP-SAT 솔버를 위한 기본 클래스.
    """

    def __init__(self, input_data):
        super().__init__(input_data)
        self.model = cp_model.CpModel()
        if not hasattr(self.model, 'named_constraints'):
            self.model.named_constraints = {}
        # CP-SAT에서는 solver 객체를 solve 직전에 생성합니다.

    def _extract_results(self, solver):
        # CP-SAT의 결과 추출은 solver 객체가 필요하므로, 인자를 받도록 재정의합니다.
        raise NotImplementedError

    @log_solver_solve
    def solve(self):
        try:
            self._create_variables()
            self._add_constraints()
            self._set_objective_function()

            solver = cp_model.CpSolver()
            # solver.parameters.max_time_in_seconds = 30.0 # 필요시 시간 제한 설정
            export_cp_model(self.model, f'ortools_{self.problem_type}.mps')
            status = solver.Solve(self.model)
            processing_time = self.get_time(solver.WallTime())
            self.log_solve_resulte(solver.StatusName(status), processing_time)

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                if status == status == cp_model.FEASIBLE:
                    msg = "Feasible solution found, but it might not be optimal."
                    logger.warning(msg)

                results = self._extract_results(solver)  # solver 객체를 전달
                error_msg = None
                return results, error_msg, processing_time
            else:
                error_msg = f"Optimal solution not found. Solver status: {solver.StatusName(status)}"
                return None, error_msg, processing_time

        except Exception as e:
            return None, e, None


class BaseOrtoolsRoutingSolver(BaseSolver):
    """
    OR-Tools의 라우팅 라이브러리(pywrapcp)를 위한 기본 클래스.
    """

    def __init__(self, input_data):
        super().__init__(input_data)
        # 라우팅 모델은 자식 클래스에서 데이터를 파싱한 후 생성합니다.
        self.manager = None
        self.routing = None
        self.solution = None
        self.status_map = {
            routing_enums_pb2.RoutingSearchStatus.ROUTING_NOT_SOLVED: "NOT_SOLVED",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS: "SUCCESS",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED: "PARTIAL_SUCCESS",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL: "FAIL",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL_TIMEOUT: "TIMEOUT",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_INVALID: "INVALID",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_INFEASIBLE: "INFEASIBLE",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL: "OPTIMAL",
        }

    def _compute_distance_matrix(self):
        """위치 좌표를 기반으로 유클리드 거리 행렬을 계산합니다."""
        num_locations = len(self.locations)
        matrix = [[0] * num_locations for _ in range(num_locations)]
        for from_node in range(num_locations):
            for to_node in range(num_locations):
                if from_node != to_node:
                    loc1 = self.locations[from_node]
                    loc2 = self.locations[to_node]
                    dist = sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
                    matrix[from_node][to_node] = int(round(dist*100))
        return matrix

    def _extract_results(self):
        # 라우팅 결과 추출은 매우 유사하므로 부모에서 일부 공통 로직을 제공할 수 있습니다.
        raise NotImplementedError

    @log_solver_solve
    def solve(self):
        try:
            self._create_variables()  # VRP에서는 manager, routing 객체 생성에 해당
            self._add_constraints()  # 거리/수요 콜백 등록, 제약 추가 등
            self._set_objective_function()  # 비용 평가자(Cost Evaluator) 설정
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            search_parameters.time_limit.FromSeconds(10)  # 타임아웃

            # ---
            start_time = time.time()
            self.solution = self.routing.SolveWithParameters(search_parameters)
            end_time = time.time()
            processing_time = self.get_time(end_time - start_time)
            status_code = self.routing.status()
            status_name = self.status_map.get(status_code, "UNKNOWN")
            self.log_solve_resulte(status_name, processing_time)

            if self.solution and status_code in [routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS,
                                                 routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL]:
                results = self._extract_results()
                return results, None, processing_time
            else:
                error_msg = f"최적 경로를 찾지 못했습니다. (상태: {status_name})"
                return None, error_msg, processing_time
        except Exception as e:
            logger.error(f"An unexpected error occurred in {self.__class__.__name__}: {e}", exc_info=True)
            return None, f"오류 발생: {str(e)}", 0.0