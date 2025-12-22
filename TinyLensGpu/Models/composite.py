"""
Composite physical model .

This module provides the PhysicalModel class that combines mass and light
profiles into a complete gravitational lensing model using the caskade framework.
"""

import caskade as ck
from typing import List, Optional


class PhysicalModel(ck.Module):
    """
    Composite physical model combining mass and light profiles.

    This class serves as a container for all physical components of a
    gravitational lens system:
    - lens_mass: Mass distribution profiles (e.g., SIE, Shear)
    - source_light: Source galaxy light profiles
    - lens_light: Lens galaxy light profiles

    All components are automatically registered as sub-modules, enabling
    caskade's automatic parameter management and forward computation.

    Parameters
    ----------
    lens_mass : list of ck.Module, optional
        List of mass profile modules (default: empty list)
    source_light : list of ck.Module, optional
        List of source light profile modules (default: empty list)
    lens_light : list of ck.Module, optional
        List of lens light profile modules (default: empty list)

    Examples
    --------
    >>> from TinyLensGpu.Models.mass import SIE, Shear
    >>> from TinyLensGpu.Models.light import SersicEllipse
    >>>
    >>> # Create individual components
    >>> sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    >>> shear = Shear(gamma1=0.05, gamma2=0.05)
    >>> source = SersicEllipse(R_sersic=0.3, n_sersic=1.0, ...)
    >>> lens = SersicEllipse(R_sersic=1.0, n_sersic=4.0, ...)
    >>>
    >>> # Combine into physical model
    >>> phys_model = PhysicalModel(
    ...     lens_mass=[sie, shear],
    ...     source_light=[source],
    ...     lens_light=[lens]
    ... )
    """

    def __init__(
        self,
        lens_mass: Optional[List[ck.Module]] = None,
        source_light: Optional[List[ck.Module]] = None,
        lens_light: Optional[List[ck.Module]] = None,
    ):
        super().__init__()

        # Store component lists as regular Python lists (not caskade NodeList)
        # IMPORTANT: Use object.__setattr__ to bypass caskade's __setattr__ interception.
        # If we use normal assignment (self._lens_mass_list = ...), caskade will try to
        # convert the list to a NodeList, which fails when multiple modules have the same
        # type name (e.g., 15 "GaussianEllipse" modules in MGE), causing GraphError.
        object.__setattr__(self, '_lens_mass_list', lens_mass or [])
        object.__setattr__(self, '_source_light_list', source_light or [])
        object.__setattr__(self, '_lens_light_list', lens_light or [])

        # Register all sub-modules with unique names for caskade's parameter tracking
        # This allows caskade to automatically track all parameters in the module tree
        for i, mass in enumerate(self._lens_mass_list):
            setattr(self, f"lens_mass_{i}", mass)

        for i, light in enumerate(self._source_light_list):
            setattr(self, f"source_light_{i}", light)

        for i, light in enumerate(self._lens_light_list):
            setattr(self, f"lens_light_{i}", light)

    @property
    def lens_mass(self):
        """Access lens mass components as a list."""
        return self._lens_mass_list

    @property
    def source_light(self):
        """Access source light components as a list."""
        return self._source_light_list

    @property
    def lens_light(self):
        """Access lens light components as a list."""
        return self._lens_light_list

    @ck.forward
    def deflection(self, x, y):
        """
        Calculate total deflection from all mass components.

        The deflection angles are computed by summing contributions from
        all mass profiles and ray-tracing to the source plane.

        Parameters
        ----------
        x : array_like
            x-coordinates in the image plane
        y : array_like
            y-coordinates in the image plane

        Returns
        -------
        beta_x : array_like
            Source plane x-coordinates after ray-tracing
        beta_y : array_like
            Source plane y-coordinates after ray-tracing
        """
        beta_x, beta_y = x, y

        # Sum deflections from all mass components
        for mass_model in self.lens_mass:
            alpha_x, alpha_y = mass_model.deriv(x, y)
            beta_x = beta_x - alpha_x
            beta_y = beta_y - alpha_y

        return beta_x, beta_y

    @ck.forward
    def source_surface_brightness(self, beta_x, beta_y):
        """
        Calculate total source surface brightness.

        Parameters
        ----------
        beta_x : array_like
            Source plane x-coordinates
        beta_y : array_like
            Source plane y-coordinates

        Returns
        -------
        total_brightness : array_like
            Total surface brightness from all source light components
        """
        total = 0.0
        for light_model in self.source_light:
            total = total + light_model.light(beta_x, beta_y)
        return total

    @ck.forward
    def lens_surface_brightness(self, x, y):
        """
        Calculate total lens surface brightness.

        Parameters
        ----------
        x : array_like
            Image plane x-coordinates
        y : array_like
            Image plane y-coordinates

        Returns
        -------
        total_brightness : array_like
            Total surface brightness from all lens light components
        """
        total = 0.0
        for light_model in self.lens_light:
            total = total + light_model.light(x, y)
        return total

    def get_component_counts(self):
        """
        Get the number of components in each category.

        Returns
        -------
        dict
            Dictionary with counts of lens_mass, source_light, and lens_light
        """
        return {
            'n_lens_mass': len(self.lens_mass),
            'n_source_light': len(self.source_light),
            'n_lens_light': len(self.lens_light)
        }
