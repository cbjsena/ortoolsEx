import logging

# 이 모듈의 로거를 가져옵니다.
logger = logging.getLogger(__name__)

class BaseSolver:
    """
    모든 최적화 솔버 클래스를 위한 최상위 기본(Abstract Base) 클래스.
    모든 솔버가 따라야 할 공통적인 구조와 인터페이스를 정의합니다.
    """
    def __init__(self, input_data):
        """
        공통 초기화 로직: 입력 데이터를 저장하고 문제 유형을 식별합니다.
        """
        self.input_data = input_data
        self.problem_type = input_data.get('problem_type', 'Unknown')
        logger.info(f"Initializing {self.__class__.__name__} for '{self.problem_type}'...")

    def _create_variables(self):
        """
        결정 변수를 생성하는 추상 메서드.
        자식 클래스에서 반드시 재정의(override)해야 합니다.
        """
        raise NotImplementedError("'_create_variables' method must be implemented in the child class.")

    def _add_constraints(self):
        """
        제약 조건을 모델에 추가하는 추상 메서드.
        자식 클래스에서 반드시 재정의해야 합니다.
        """
        raise NotImplementedError("'_add_constraints' method must be implemented in the child class.")

    def _set_objective_function(self):
        """
        목표 함수를 설정하는 추상 메서드.
        자식 클래스에서 반드시 재정의해야 합니다.
        """
        raise NotImplementedError("'_set_objective_function' method must be implemented in the child class.")

    def _extract_results(self):
        """
        솔버 실행 후 결과를 추출하는 추상 메서드.
        자식 클래스에서 반드시 재정의해야 합니다.
        """
        raise NotImplementedError("'_extract_results' method must be implemented in the child class.")

    def get_time(self, processing_time:float):
        return f"{processing_time:.2f}" if processing_time is not None else "N/A"

    def log_solve_resulte(self, status_name:str, processing_time:str):
        """
        솔버 실행 결과 상태와 실행 시간 로그 출력
        """
        logger.info(f"Solver finished. Status: {status_name}, Time: {processing_time} sec")

    def solve(self):
        """
        전체 최적화 프로세스를 실행하는 메인 메서드.
        이 메서드는 모든 자식 클래스에서 공통으로 사용되는 템플릿 역할을 합니다.
        """
        try:
            # 1. 모델링 단계
            self._create_variables()
            self._add_constraints()
            self._set_objective_function()

            # 2. 해결 단계 (실제 해결 로직은 회사별 BaseSolver에서 구현)
            # 이 최상위 클래스에서는 solve 로직을 직접 구현하지 않습니다.
            raise NotImplementedError("'solve' method's core logic must be implemented in library-specific base solvers.")

        except NotImplementedError as nie:
            logger.error(f"A required method was not implemented in {self.__class__.__name__}: {nie}", exc_info=True)
            return None, f"솔버 구현 오류: {nie}", 0.0
        except Exception as e:
            logger.error(f"An unexpected error occurred in {self.__class__.__name__}: {e}", exc_info=True)
            return None, f"오류 발생: {str(e)}", 0.0
