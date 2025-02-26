from abc import ABC, abstractmethod
from typing import List


class Parameterized(ABC):
    """Interface for a parameterized profile of any kind.

    Attributes:
        name (str): Name of the profile
        params (:obj:`list` of :obj:`str`): List of parameter names
    """

    _name: str  # Static class level default for name
    _params: List[str]  # # Static class level default for parameter names

    def __init__(self, *args, **kwargs):
        self.name = self._name
        self.params = self._params

    def __str__(self):
        return self.name


class LightProfile(Parameterized, ABC):
    """Interface for a light profile.
    """
    
    @abstractmethod
    def light(self, x, y, **kwargs):
        pass


class MassProfile(Parameterized, ABC):
    """Interface for a mass profile."""

    @abstractmethod
    def deriv(self, x, y, **kwargs):
        """Calculates deflection angle.

        Args:
            x: :math:`x` coordinate at which to evaluate the deflection
            y: :math:`y` coordinate at which to evaluate the deflection
            **kwargs: Mass profile parameters. Each parameter must be shaped in a way that is broadcastable with x and y

        Returns:
            A tuple :math:`(\\alpha_x, \\alpha_y)` containing the deflection angle in the :math:`x` and :math:`y` directions

        """
        pass
