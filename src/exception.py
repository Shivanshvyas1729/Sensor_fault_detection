import sys
import traceback

class CustomException(Exception):
    def __init__(self, error, sys_info):
        self.error = error
        self.sys_info = sys_info

        _, _, exc_tb = sys_info.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        error_message = (
            f"Error occurred in script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"error message: [{error}]"
        )

        super().__init__(error_message)
