from logging_config import setup_logger
import logging
import datetime

setup_logger()
logger = logging.getLogger(__name__)

def start_log(problem_type:str):
    logger.info(f"Running Optimizer {problem_type}")

def solving_log(solver, problem_type:str, model=None):
    logger.info(f"Solving the {problem_type} model")
    if solver.__class__.__name__ == "CpSolver":
        status = solver.Solve(model)
    else:
        status = solver.Solve()
    processing_time = get_solving_time_sec(solver)
    logger.info(f"Solver finished. Status: {status}, Time: {processing_time} sec")

    return status, processing_time


def gurobi_solving_log(model, problem_type:str):
    logger.info(f"Solving the {problem_type} model")
    model.optimize()
    status = model.status
    processing_time = model.Runtime
    logger.info(f"Solver finished. Status: {status}, Time: {processing_time} sec")

    return status, processing_time


def ortools_routing_solving_log(routing, search_parameters, problem_type:str):
    logger.info(f"Solving the {problem_type} model")
    solve_start_time = datetime.datetime.now()
    solution = routing.SolveWithParameters(search_parameters)
    status = routing.status()
    solve_end_time = datetime.datetime.now()
    processing_time = (solve_end_time - solve_start_time).total_seconds()
    logger.info(f"Solver finished. Status: {status}, Time: {processing_time} sec")

    return solution, status, processing_time

def get_solving_time_sec(solver):
    """
       OR-Tools solver의 WallTime을 초 단위 문자열로 반환합니다.
       - CP-SAT solver는 초(sec)
       - Linear solver는 밀리초(ms)
       """
    try:
        time_raw = solver.WallTime()
    except AttributeError:
        return "N/A"

    # CP-SAT solver는 일반적으로 `CpSolver` 클래스의 인스턴스
    is_cp_sat = solver.__class__.__name__ == "CpSolver"

    processing_time = time_raw if is_cp_sat else time_raw / 1000  # 초 단위로 통일
    return f"{processing_time:.3f}" if processing_time is not None else "N/A"