import logging
from functools import wraps
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
import time

def log_view_activity(view_func):
    """
    뷰 함수의 시작, 종료, 예외 발생 시 자동으로 로그를 기록하는 데코레이터.
    """

    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # 뷰 함수가 정의된 모듈의 로거를 가져옵니다.
        # 이렇게 하면 settings.py에 설정된 앱별 로거가 올바르게 사용됩니다.
        logger = logging.getLogger(view_func.__module__)

        view_name = view_func.__name__
        method = request.method

        if 'introduction_view' not in view_name:
            logger.info(f"[{view_name}] {method} request received.")
        if method == 'POST':
            # POST 데이터는 민감할 수 있으므로 DEBUG 레벨로 기록합니다.
            logger.debug(f"[{view_name}] Form data: {request.POST}")

        try:
            # 원래의 뷰 함수를 실행합니다.
            response = view_func(request, *args, **kwargs)

            # 뷰가 TemplateResponse를 반환하고 context 데이터를 가지고 있는지 확인합니다.
            error_message_in_context = None
            if hasattr(response, 'context_data'):
                error_message_in_context = response.context_data.get('error_message')

            # context에 error_message가 설정되어 있지 않은 경우에만 성공 로그를 기록합니다.
            if not error_message_in_context:
                logger.info(f"[{view_name}] View executed successfully and rendered.")
            else:
                # 에러 메시지가 있다면, 이는 '관리된 오류'이므로 WARNING 레벨로 기록합니다.
                logger.warning(f"[{view_name}] View executed with a controlled error message.")
            return response

        except Exception as e:
            # 뷰 함수 실행 중 예외가 발생하면 에러 로그를 남깁니다.
            logger.error(f"[{view_name}] An error occurred: {e}", exc_info=True)
            # 예외를 다시 발생시켜 Django의 기본 에러 처리 메커니즘이 동작하도록 합니다.
            raise

    return wrapper


def log_data_creation(func):
    """
    데이터 생성 함수의 시작, 성공, 실패를 자동으로 로깅하는 데코레이터.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 함수가 정의된 모듈의 로거를 사용합니다.
        logger = logging.getLogger(func.__module__)
        func_name = func.__name__

        logger.info(f"[{func_name}] Starting data creation...")

        try:
            # 원래의 데이터 생성 함수를 실행합니다.
            returned_value = func(*args, **kwargs)

            # 반환 값의 형태에 따라 로깅할 데이터를 결정합니다. ---
            result_data_for_logging = None

            # 1. 반환 값이 두 개의 항목을 가진 튜플인 경우 (message, result_data 형태)
            if isinstance(returned_value, tuple) and len(returned_value) == 2:
                # 로깅 및 데이터 존재 여부 확인은 두 번째 항목을 기준으로 합니다.
                result_data_for_logging = returned_value[1]
            else:
                # 2. 그 외의 경우 (단일 값 반환)
                result_data_for_logging = returned_value

            if result_data_for_logging:
                if logger.isEnabledFor(logging.DEBUG):
                    for key, value in result_data_for_logging.items():
                        logger.debug(f"[{key}]: {value}")
                logger.info(f"[{func_name}] Data created successfully.")
            else:
                logger.warning(f"[{func_name}] Function executed but returned no data (None or empty).")

            return returned_value

        except ValueError as ve:
            # 입력값 오류는 WARNING 레벨로 기록
            logger.warning(f"[{func_name}] Input validation error: {ve}")
            # 예외를 다시 발생시켜 뷰에서 처리하도록 함
            raise
        except Exception as e:
            # 그 외 예상치 못한 오류는 ERROR 레벨로 기록
            logger.error(f"[{func_name}] An unexpected error occurred: {e}", exc_info=True)
            raise

    return wrapper

def log_solver_make(func):
    """
    최적화 솔버의 모델 생성 함수들의 시작, 종료, 상태, 처리 시간을 자동으로 로깅하는 데코레이터.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 함수가 정의된 모듈의 로거를 사용
        instance = args[0]
        logger = logging.getLogger(func.__module__)
        func_name = func.__name__

        # input_data에서 problem_type 추출 시도
        problem_type = getattr(instance, 'problem_type', 'Unknown')

        logger.solve(f"Starting [{func_name}] in solver for '{problem_type}'...")
        start_time = time.time()

        try:
            # 원래의 솔버 함수 실행
            func(*args, **kwargs)  # 처리 시간은 데코레이터가 계산
            end_time = time.time()
            processing_time = round(end_time - start_time, 4)
            logger.solve(f"Ended [{func_name}] in solver for '{problem_type}', Time: {processing_time} sec")
        except Exception as e:
            end_time = time.time()
            processing_time = round(end_time - start_time, 4)
            logger.error(
                f"[{func_name}] An unexpected error occurred in solver for '{problem_type}': {e}. Time: {processing_time} sec",
                exc_info=True)

    return wrapper

def log_solver_solve(func):
    """
    최적화 솔버 클래스의 solve 함수의 시작, 종료, 상태, 처리 시간을 자동으로 로깅하는 데코레이터.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # args[0]은 클래스 인스턴스(self)입니다.
        # 인스턴스에서 로거와 problem_type을 가져옵니다.
        instance = args[0]
        logger = logging.getLogger(func.__module__)
        func_name = func.__name__

        # input_data에서 problem_type 추출 시도
        problem_type = getattr(instance, 'problem_type', 'Unknown')

        logger.info(f"[{func_name}] Starting solver for '{problem_type}'...")
        start_time = time.time()

        try:
            # 원래의 솔버 함수 실행
            results, error_msg, _ = func(*args, **kwargs)  # 처리 시간은 데코레이터가 계산

            end_time = time.time()
            processing_time = round(end_time - start_time, 4)

            if error_msg:
                logger.warning(
                    f"[{func_name}] Solver finished for '{problem_type}' with a message: {error_msg}. Time: {processing_time} sec")
            else:
                logger.info(
                    f"[{func_name}] Solver finished successfully for '{problem_type}'. Time: {processing_time} sec")

            return results, error_msg, processing_time

        except Exception as e:
            end_time = time.time()
            processing_time = round(end_time - start_time, 4)
            logger.error(
                f"[{func_name}] An unexpected error occurred in solver for '{problem_type}': {e}. Time: {processing_time} sec",
                exc_info=True)
            # 뷰로 전달할 에러 메시지와 함께 예외를 다시 발생시키거나, 튜플을 반환할 수 있음
            return None, f"솔버 실행 중 오류 발생: {str(e)}", processing_time

    return wrapper