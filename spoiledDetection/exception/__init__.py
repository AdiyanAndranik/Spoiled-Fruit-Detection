import sys
import os

def error_message_detail(error, error_detail):
    if error_detail is None or not hasattr(error_detail, '__getitem__'):
        return f"Error: {str(error)}"
    
    _, _, exc_tb = error_detail
    if exc_tb is None:
        return f"Error: {str(error)}"
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"
    return error_message

class AppException(Exception):
    def __init__(self, error_message, error_detail):
        self.error_message = error_message_detail(error_message, error_detail)
        super().__init__(self.error_message)