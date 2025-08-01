class UntestedError(Exception):
    """
    This code is not tested yet, please verify!
    """
    def __init__(self, message="This code is not tested yet, please verify!"):
        super().__init__(message)
        

    