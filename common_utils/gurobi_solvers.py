import logging
import gurobipy as gp
from gurobipy import GRB
from .base_solver import BaseSolver

logger = logging.getLogger(__name__)


class BaseGurobiSolver(BaseSolver):
    """
    Gurobi 솔버를 위한 기본 클래스.
    gurobipy 라이브러리의 API를 사용합니다.
    """

    def __init__(self, input_data):
        super().__init__(input_data)
        # Gurobi는 Model 객체를 먼저 생성합니다.
        self.model = gp.Model(self.problem_type)
        self.status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
        }

    def _extract_results(self):
        # Gurobi의 결과 추출은 self.model의 상태를 확인하므로, 인자가 필요 없습니다.
        raise NotImplementedError

    def solve(self):
        try:
            self._create_variables()
            self._add_constraints()
            self._set_objective_function()

            # Gurobi의 해결(optimize) 메서드 호출
            self.model.optimize()

            status_code = self.model.Status
            status_name = self.status_map.get(status_code, f"UNKNOWN_STATUS_{status_code}")
            processing_time = self.model.Runtime
            self.log_solve_resulte(status_name, processing_time)

            if status_code in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                results = self._extract_results()
                error_msg = None
                if status_code != GRB.OPTIMAL:
                    error_msg = f"최적해는 아니지만 실행 가능한 해를 찾았습니다. (상태: {status_name})"

                return results, error_msg, processing_time
            else:
                error_msg = f"Optimal solution not found. Gurobi status: {status_name}"
                return None, error_msg, processing_time

        except Exception as e:
            # BaseSolver의 에러 처리 로직을 그대로 활용
            return super().solve()
