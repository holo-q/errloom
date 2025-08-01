from enum import Enum

class DeploymentStep(Enum):
    """
    Refers to the progress of a deployment installation. (cloning, apt packages, pip installs)
    """

    ZERO = 0
    GIT = 1
    DONE = 2