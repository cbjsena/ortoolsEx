SOLVER_ORTOOLS = 'ORTOOLS'
ORTOOLS_TIME_LIMIT=180
SOLVER_GUROBI = 'GUROBI'
GUROBI_TIME_LIMIT=60

def get_solving_time_sec(processing_time):
    # solver.WallTime(): if solver is CP-SAT then, sec else ms
    processing_time = processing_time / 1000
    return f"{processing_time:.3f}" if processing_time is not None else "N/A"


def get_solving_time_cp_sec(processing_time):
    # solver.WallTime(): if solver is CP-SAT then, sec else ms
    processing_time = processing_time
    return f"{processing_time:.3f}" if processing_time is not None else "N/A"