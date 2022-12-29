from typing import TypeVar

import numpy as np
import numpy.typing as npt

DTypeT = TypeVar("DTypeT", bound=np.generic, covariant=True)


def split_by_cond(array: npt.NDArray[DTypeT],
                  cond: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[DTypeT], npt.NDArray[DTypeT]]:
    """Split <array> into two NDArrays, one that satisfies <cond> and one that does not.

    Args:
        array (npt.NDArray[DTypeT]): The NDArray to be split.
        cond (npt.NDArray[np.bool_]): The NDArray of Boolean type used as a criteria for splitting <array>.

    Returns:
        tuple[npt.NDArray[DTypeT], npt.NDArray[DTypeT]]: The tuple of NDArrays obtained by splitting <array>.
            The first element satisfies <cond> and the second element does not.
    """
    return (array[cond], array[~cond])
