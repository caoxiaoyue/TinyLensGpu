import numpy as np
import scipy.special as sf
import matplotlib.pyplot as plt

def mag2cps(magnitude, magnitude_zero_point):
    """
    From lenstronomy

    converts an apparent magnitude to counts per second

    The zero point of an instrument, by definition, is the magnitude of an object that produces one count
    (or data number, DN) per second. The magnitude of an arbitrary object producing DN counts in an observation of
    length EXPTIME is therefore:
    m = -2.5 x log10(DN / EXPTIME) + ZEROPOINT

    :param magnitude:
    :param magnitude_zero_point:
    :return: counts per second
    """
    delta_M = magnitude - magnitude_zero_point
    cps = 10**(-delta_M/2.5) #cps: counts per second
    return cps

def cps2mag(cps, magnitude_zero_point):
    delta_M = -2.5 * np.log10(cps) #cps: counts per second
    return delta_M + magnitude_zero_point


class SersicLight(object):
    def __init__(
        self,
        xc=None, 
        yc=None, 
        q=None, 
        PA=None, 
        Re=None, 
        Ie=None, 
        n=None,
        m=None,
        mag_zero=22,
    ):
        self.xc = xc
        self.yc = yc
        self.q = q
        self.PA = PA
        self.Re = Re
        self.Ie = Ie #Note, we set the unit of Ie to counts/s/arcsec^2, not the counts/s/pixel.
        self.n = n
        self.m = m #apprent magnitude of source
        self.m0 = mag_zero

        if self.Ie is None:
            if self.m is not None:
                self.Ie = self.Ie_from_magnitude(
                    self.m, 
                    self.m0,
                    Re=self.Re,
                    n=self.n, 
                )
                self.total_flux = mag2cps(self.m, self.m0) #unit: counts/s
            else:
                raise Exception('please input either the intensity at effective radius, or magnitude')
        else:
            self.total_flux = self.total_flux_analytic_from(
                Re=self.Re, 
                Ie=self.Ie, 
                n=self.n,
            ) #unit: counts/s


    @staticmethod
    def total_flux_analytic_from(
        Re=None,
        Ie=None,
        n=None, 
    ):
        #result is almost the same as the `total_flux_from`, but faster speed
        k = sf.gammaincinv(2.0*n, 0.5)
        factor = k**(2.0*n) / (2*np.pi*Re**2*np.exp(k)*n*sf.gamma(2.0*n))
        return Ie/factor


    @staticmethod
    def Ie_from_magnitude(
        magnitude, 
        magnitude_zero_point=22.0,
        Re=None,
        n=None, 
    ):
        total_cps_tmp = SersicLight.total_flux_analytic_from(Re=Re, Ie=1.0, n=n)
        total_cps = mag2cps(magnitude, magnitude_zero_point)
        rescale_factor = total_cps/total_cps_tmp
        return rescale_factor
    

def compute_nmad_and_outliers(z_true, z_pred, threshold=1, normalize=True, no_one=False):
    """
    Compute NMAD and catastrophic outlier fraction.

    Parameters:
        z_true (array-like): True redshifts.
        z_pred (array-like): Predicted redshifts.
        threshold (float): Threshold for catastrophic outliers in normalized residuals (in many sigma).
        normalize (bool): Whether to normalize the residuals.
        no_one (bool): Whether to not normalize by (1+z_true).
    Returns:
        sigma_nmad (float): Normalized Median Absolute Deviation (NMAD).
        outlier_fraction (float): Fraction of catastrophic outliers.
        delta_z (array): Normalized residuals.
    """
    if normalize:
        if no_one:
            delta_z = (z_pred - z_true) / z_true
        else:
            delta_z = (z_pred - z_true) / (1 + z_true)
    else:
        delta_z = z_pred - z_true
    median_delta = np.median(delta_z)
    mad = np.median(np.abs(delta_z - median_delta))
    sigma_nmad = 1.48 * mad  # NMAD definition
    outlier_fraction = np.sum(np.abs(delta_z) > threshold * sigma_nmad) / len(z_true) * 100  # Percentage of catastrophic outliers

    true_range = np.max(z_true) - np.min(z_true)
    true_bound = np.array([np.min(z_true)-true_range/2, np.max(z_true)+true_range/2])
    if normalize:
        if no_one:
            diff = threshold * sigma_nmad * true_bound
        else:
            diff = threshold * sigma_nmad * (1+true_bound)
    else:
        diff = threshold * sigma_nmad
    pred_upper_bound = true_bound + diff
    pred_lower_bound = true_bound - diff

    bound_all = [true_bound, pred_upper_bound, pred_lower_bound]

    return sigma_nmad, outlier_fraction, delta_z, bound_all


def nmad_plot(ax_main, ax_residual, z_true, z_m, z_m_err, threshold=5, normalize=True):
    sigma_nmad, outlier_fraction, delta_z, bound_all = compute_nmad_and_outliers(z_true, z_m, threshold=threshold, normalize=normalize)
    #also plot the error bar
    ax_main.errorbar(z_true, z_m, yerr=z_m_err, fmt="o", color="black", alpha=0.5, rasterized=True, 
                    markersize=2, elinewidth=0.8, capsize=2, capthick=0.8)
    ax_main.plot(bound_all[0], bound_all[0], color="red", linestyle="-", linewidth=1) # 1:1 line
    ax_main.plot(bound_all[0], bound_all[1], color="red", linestyle="--", linewidth=1) # upper bound
    ax_main.plot(bound_all[0], bound_all[2], color="red", linestyle="--", linewidth=1) # lower bound

    stats_text = rf"$\sigma={sigma_nmad:.3f}$" + "\n" + r"$f_c=$" + rf"${outlier_fraction:.2f}\%$"
    ax_main.text(0.05, 0.8, stats_text, transform=ax_main.transAxes, fontsize=10)
    # ax_main.set_xticklabels([])
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_residual.get_xticklabels(), visible=True)

    ax_residual.scatter(z_true, delta_z, s=2, color="black", alpha=0.5, rasterized=True)
    ax_residual.axhline(y=0, color="red", linestyle="-", linewidth=1)
    ax_residual.axhline(y=threshold*sigma_nmad, color="red", linestyle="--", linewidth=1)
    ax_residual.axhline(y=-threshold*sigma_nmad, color="red", linestyle="--", linewidth=1)


def nmad_plot_single_truth(ax_main, ax_residual, z_true, z_m, z_m_err, threshold=5, normalize=True):
    z_true = np.full_like(z_m, z_true)
    sigma_nmad, outlier_fraction, delta_z, bound_all = compute_nmad_and_outliers(z_true, z_m, threshold=threshold, normalize=normalize)
    #also plot the error bar
    ax_main.errorbar(np.arange(len(z_m))+1, z_m, yerr=z_m_err, fmt="o", color="black", alpha=0.5, 
                    rasterized=True, markersize=2, elinewidth=0.8, capsize=2, capthick=0.8)
    ax_main.axhline(y=z_true[0], color="black", linestyle="--", linewidth=1)
    ax_main.axhline(y=bound_all[1][0], color="red", linestyle="--", linewidth=1)
    ax_main.axhline(y=bound_all[2][0], color="red", linestyle="--", linewidth=1)

    stats_text = rf"$\sigma={sigma_nmad:.3f}$" + "\n" + r"$f_c=$" + rf"${outlier_fraction:.2f}\%$"
    ax_main.text(0.05, 0.8, stats_text, transform=ax_main.transAxes, fontsize=10)
    # ax_main.set_xticklabels([])
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_residual.get_xticklabels(), visible=True)

    ax_residual.scatter(np.arange(len(z_m))+1, delta_z, s=2, color="black", alpha=0.5)
    ax_residual.axhline(y=0, color="red", linestyle="-", linewidth=1)
    ax_residual.axhline(y=threshold*sigma_nmad, color="red", linestyle="--", linewidth=1)
    ax_residual.axhline(y=-threshold*sigma_nmad, color="red", linestyle="--", linewidth=1)