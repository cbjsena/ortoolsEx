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

            processing_time = self.model.Runtime

            # Gurobi의 상태(status) 확인
            if self.model.Status == GRB.OPTIMAL:
                results = self._extract_results()
                return results, None, processing_time
            elif self.model.Status == GRB.SUBOPTIMAL:
                results = self._extract_results()
                logger.warning(f"Suboptimal solution found for {self.problem_type}.")
                return results, "최적해는 아니지만 실행 가능한 해를 찾았습니다.", processing_time
            else:
                error_msg = f"Optimal solution not found. Gurobi status code: {self.model.Status}"
                return None, error_msg, processing_time

        except Exception as e:
            # BaseSolver의 에러 처리 로직을 그대로 활용
            return super().solve()
