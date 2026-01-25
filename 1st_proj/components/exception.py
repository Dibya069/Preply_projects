## To handle the Error gracefully

import sys
from components.logger import logger

def error_msg_datils(error, error_detail: sys):
    ## Generate the detail error message with file name, line number.

    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = f"Error occured in file: [{file_name}] at line number [{line_number}] error message [{str(error)}]"
    logger.info(error_message)
    return error_message
    

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_msg_datils(error_message, error_detail)


    def __str__(self):
        return self.error_message
    



"""
implementation of the custom exception class
from components.exception import CustomException

try:
    a = 1/0
except Exception as e:
    raise CustomException(e, sys)

"""