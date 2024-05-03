from typing import Literal


class Params:
    """
    The different parametrizations that are allowed for the stable
    distribution:

    - N0: The main parametrization. It is the most favorable from a numerical
    perspective (the pdf is continuous in the parameters).

    - N1: The parametrization used by scipy by default. The most intuitive
    for users, since the `loc` parameter is equal to the mean of the distribution
    in this parametrization.

    """

    N0: Literal["N0"] = "N0"
    N1: Literal["N1"] = "N1"


Param = Literal["N0", "N1"]
