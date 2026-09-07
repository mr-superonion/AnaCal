"""COSMOS-based galaxy image simulation (GalSim), for test fixtures.

This used to be ``anacal.simulation``.  AnaCal itself no longer depends
on GalSim; the images the tests and examples need are rendered once by
``make_fixtures.py`` in this directory and stored as FITS, and this
module is the renderer that script uses.  It is not part of the
installed package.
"""
# FPFS shear estimator
# Copyright 20210905 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# python lib

import gc
import os

import fitsio
import galsim
import numpy as np

_data_dir = os.path.dirname(os.path.abspath(__file__))


def _galsim_round_sersic(n, sersic_prec):
    """Round a Sersic index to the nearest multiple of *sersic_prec*."""
    return float(int(n / sersic_prec + 0.5)) * sersic_prec


def coord_rotate(x, y, xref, yref, theta):
    """Rotate coordinates around a reference point.

    Args:
        x (np.ndarray): x-coordinates in pixels.
        y (np.ndarray): y-coordinates in pixels.
        xref (float): Reference x-coordinate in pixels.
        yref (float): Reference y-coordinate in pixels.
        theta (float): Rotation angle in radians, positive anticlockwise.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotated ``x`` and ``y`` coordinates in
        pixels.

    Notes:
        This function returns new arrays and does not modify ``x`` or ``y``.

    References:
        Rotation matrix: https://en.wikipedia.org/wiki/Rotation_matrix
    """
    xu = x - xref
    yu = y - yref
    x2 = np.cos(theta) * xu - np.sin(theta) * yu + xref
    y2 = np.sin(theta) * xu + np.cos(theta) * yu + yref
    return x2, y2


def generate_cosmos_gal_simple(record, gsparams=None, random_shape=False):
    """Generate a simplified COSMOS galaxy (Gaussian profiles only).

    Uses Gaussian approximations for both bulge+disk and single-Sersic
    fits, which is faster than :func:`generate_cosmos_gal` and suitable
    for use with :class:`galsim.RandomKnots`.

    Args:
        record (ndarray): One row of the COSMOS galaxy catalogue.
        gsparams (galsim.GSParams | None): GalSim rendering parameters.
        random_shape (bool): Whether to apply intrinsic ellipticity
            from the catalogue fit parameters.

    Returns:
        galsim.GSObject: GalSim galaxy object.
    """

    bdparams = record["bulgefit"]
    sparams = record["sersicfit"]
    use_bulgefit = record["use_bulgefit"]
    if use_bulgefit:
        # Bulge parameters:
        # Minor-to-major axis ratio:
        bulge_hlr = record["hlr"][1]
        bulge_flux = record["flux"][1]
        disk_hlr = record["hlr"][2]
        disk_flux = record["flux"][2]
        bulge = galsim.Gaussian(
            flux=bulge_flux, half_light_radius=bulge_hlr, gsparams=gsparams
        )
        disk = galsim.Exponential(
            flux=disk_flux, half_light_radius=disk_hlr, gsparams=gsparams
        )
        if random_shape:
            # Apply shears for intrinsic shape.
            bulge_q = bdparams[11]
            bulge_beta = bdparams[15] * galsim.radians
            if bulge_q < 1.0:  # pragma: no branch
                bulge = bulge.shear(q=bulge_q, beta=bulge_beta)
            #
            disk_q = bdparams[3]
            disk_beta = bdparams[7] * galsim.radians
            if disk_q < 1.0:  # pragma: no branch
                disk = disk.shear(q=disk_q, beta=disk_beta)
        # Then combine the two components of the galaxy.
        # No center offset is included
        gal = bulge + disk
    else:
        gal_hlr = record["hlr"][0]
        gal_flux = record["flux"][0]

        gal = galsim.Gaussian(
            flux=gal_flux,
            half_light_radius=gal_hlr,
        )
        if random_shape:
            # Apply shears for intrinsic shape.
            gal_q = sparams[3]
            gal_beta = sparams[7] * galsim.radians
            if gal_q < 1.0:
                gal = gal.shear(q=gal_q, beta=gal_beta)
    return gal


def generate_cosmos_gal(record, trunc_ratio=5.0, gsparams=None):
    """Generate a realistic COSMOS galaxy with Sersic/DeVaucouleurs profiles.

    Builds either a bulge+disk model (DeVaucouleurs bulge + Exponential
    disk) or a single Sersic profile, applying intrinsic ellipticity from
    the catalogue fit parameters.  Profiles are optionally truncated at
    ``trunc_ratio * half_light_radius``.

    Based on the GalSim COSMOS scene implementation.

    Args:
        record (ndarray): One row of the COSMOS galaxy catalogue,
            containing ``bulgefit``, ``sersicfit``, ``use_bulgefit``,
            ``hlr``, and ``flux`` fields.
        trunc_ratio (float): Truncation radius in units of the
            half-light radius.  Set to < 1 to disable truncation.
        gsparams (galsim.GSParams | None): GalSim rendering parameters.

    Returns:
        galsim.GSObject: GalSim galaxy object with intrinsic shape
        applied.
    """

    # record columns:
    # For 'sersicfit', the result is an array of 8 numbers for each:
    #     SERSICFIT[0]: intensity of light profile at the half-light radius.
    #     SERSICFIT[1]: half-light radius measured along the major axis, in
    #                   units of pixels in the COSMOS lensing data reductions
    #                   (0.03 arcsec).
    #     SERSICFIT[2]: Sersic n.
    #     SERSICFIT[3]: q, the ratio of minor axis to major axis length.
    #     SERSICFIT[4]: boxiness, currently fixed to 0, meaning isophotes are
    #                   all elliptical.
    #     SERSICFIT[5]: x0, the central x position in pixels.
    #     SERSICFIT[6]: y0, the central y position in pixels.
    #     SERSICFIT[7]: phi, the position angle in radians. If phi=0, the major
    #                   axis is lined up with the x axis of the image.
    # For 'bulgefit', the result is an array of 16 parameters that comes from
    # doing a 2-component sersic fit.  The first 8 are the parameters for the
    # disk, with n=1, and the last 8 are for the bulge, with n=4.

    bdparams = record["bulgefit"]
    sparams = record["sersicfit"]
    use_bulgefit = record["use_bulgefit"]
    if use_bulgefit:
        # Bulge parameters:
        # Minor-to-major axis ratio:
        bulge_hlr = record["hlr"][1]
        bulge_flux = record["flux"][1]
        disk_hlr = record["hlr"][2]
        disk_flux = record["flux"][2]
        if trunc_ratio <= 0.99:
            btrunc = None
            bulge = galsim.DeVaucouleurs(
                flux=bulge_flux, half_light_radius=bulge_hlr, gsparams=gsparams
            )
            disk = galsim.Exponential(
                flux=disk_flux, half_light_radius=disk_hlr, gsparams=gsparams
            )
        else:
            btrunc = bulge_hlr * trunc_ratio
            bulge = galsim.DeVaucouleurs(
                flux=bulge_flux,
                half_light_radius=bulge_hlr,
                trunc=btrunc,
                gsparams=gsparams,
            )
            dtrunc = disk_hlr * trunc_ratio
            disk = galsim.Sersic(
                1.0,
                flux=disk_flux,
                half_light_radius=disk_hlr,
                trunc=dtrunc,
                gsparams=gsparams,
            )
        # Apply shears for intrinsic shape.
        bulge_q = bdparams[11]
        bulge_beta = bdparams[15] * galsim.radians
        if bulge_q < 1.0:  # pragma: no branch
            bulge = bulge.shear(q=bulge_q, beta=bulge_beta)
        #
        disk_q = bdparams[3]
        disk_beta = bdparams[7] * galsim.radians
        if disk_q < 1.0:  # pragma: no branch
            disk = disk.shear(q=disk_q, beta=disk_beta)
        # Then combine the two components of the galaxy.
        # No center offset is included
        gal = bulge + disk
    else:
        # Do a similar manipulation to the stored quantities for the single
        # Sersic profiles.
        gal_n = sparams[2]
        # Fudge this if it is at the edge of the allowed n values.  Since
        # GalSim (as of #325 and #449) allow Sersic n in the range 0.3<=n<=6,
        # the only problem is that the fits occasionally go as low as n=0.2.
        # The fits in this file only go to n=6, so there is no issue with
        # too-high values, but we also put a guard on that side in case other
        # samples are swapped in that go to higher value of sersic n.
        if gal_n < 0.3:
            gal_n = 0.3
        if gal_n > 6.0:
            gal_n = 6.0

        # GalSim is much more efficient if only a finite number of Sersic n
        # values are used. This (optionally given constructor args) rounds n to
        # the nearest 0.05. change to 0.1 to speed up
        gal_n = _galsim_round_sersic(gal_n, 0.1)
        gal_hlr = record["hlr"][0]
        gal_flux = record["flux"][0]

        if trunc_ratio <= 0.99:
            strunc = None
            gal = galsim.Sersic(
                gal_n,
                flux=gal_flux,
                half_light_radius=gal_hlr,
                gsparams=gsparams,
            )
        else:
            strunc = gal_hlr * trunc_ratio
            gal = galsim.Sersic(
                gal_n,
                flux=gal_flux,
                half_light_radius=gal_hlr,
                trunc=strunc,
                gsparams=gsparams,
            )
        # Apply shears for intrinsic shape.
        gal_q = sparams[3]
        gal_beta = sparams[7] * galsim.radians
        if gal_q < 1.0:
            gal = gal.shear(q=gal_q, beta=gal_beta)
    return gal


def _generate_gal_fft(record, mag_zero, rng, gsparams):
    """Create a COSMOS galaxy with random rescaling and rotation (FFT path)."""
    gal0 = generate_cosmos_gal(record, gsparams=gsparams)
    # E.g., HSC's i-band coadds zero point is 27
    flux = 10 ** ((mag_zero - record["mag_auto"]) / 2.5)
    gal0 = gal0.withFlux(flux)
    # rescale the radius by 'rescale' and keep surface brightness the
    # same
    rescale = rng.np.uniform(0.95, 1.05)
    gal0 = gal0.expand(rescale)
    # rotate by a random angle
    ang = (rng.np.uniform(0.0, np.pi * 2.0)) * galsim.radians
    gal0 = gal0.rotate(ang)
    return gal0


def _generate_gal_mc(record, mag_zero, rng, gsparams, npoints):
    """Create a COSMOS galaxy as RandomKnots (Monte-Carlo path)."""
    # need to truncate the profile since we do not want
    # Knots locate at infinity
    galp = generate_cosmos_gal_simple(record)
    # accounting for zeropoint difference between COSMOS HST and HSC
    # HSC's i-band coadds zero point is 27
    flux = 10 ** ((mag_zero - record["mag_auto"]) / 2.5)
    galp = galp.withFlux(flux)
    # rescale the radius by 'rescale' and keep surface brightness the
    # same
    rescale = rng.np.uniform(0.95, 1.05)
    galp = galp.expand(rescale)
    # rotate by a random angle
    ang = (rng.np.uniform(0.0, np.pi * 2.0)) * galsim.radians
    galp = galp.rotate(ang)
    gal0 = galsim.RandomKnots(
        npoints=npoints,
        profile=galp,
        rng=rng,
        gsparams=gsparams,
    )
    return gal0


def make_exposure_stamp(
    sim_method,
    rng,
    mag_zero,
    psf_obj,
    scale,
    cat_input,
    ngalx,
    ngaly,
    ngrid,
    rot_field,
    g1,
    g2,
    nrot_per_gal,
    do_shift,
    buff=0,
    draw_method="auto",
):
    """Render galaxies on a grid to build exposure stamps.

    Places galaxies on a regular grid, applies shear, convolves with the
    PSF, and draws images for each field rotation angle.

    Args:
        sim_method (str): Galaxy rendering method (``"fft"`` or ``"mc"``).
        rng (galsim.BaseDeviate): GalSim random number generator.
        mag_zero (float): Magnitude zero-point.
        psf_obj (galsim.GSObject): PSF object for convolution.
        scale (float): Pixel scale in arcseconds.
        cat_input (ndarray): COSMOS galaxy catalogue rows.
        ngalx (int): Number of galaxies along the x-axis.
        ngaly (int): Number of galaxies along the y-axis.
        ngrid (int): Stamp size in pixels per galaxy.
        rot_field (list[float]): Field rotation angles in radians.
        g1 (float): First shear component.
        g2 (float): Second shear component.
        nrot_per_gal (int): Number of shape-noise-cancelling rotations
            per galaxy.
        do_shift (bool): Whether to apply random sub-pixel shifts.
        buff (int): Zero-padding buffer around the image boundary.
        draw_method (str): GalSim drawing method.

    Returns:
        list[NDArray]: List of exposure arrays, one per field rotation.
    """
    ngal = ngalx * ngaly
    gsparams = galsim.GSParams(maximum_fft_size=10240)
    gal0 = None
    gal_image_list = []
    gal_input_list = []
    npix_x = ngalx * ngrid + buff * 2.0
    npix_y = ngaly * ngrid + buff * 2.0
    for _ in range(len(rot_field)):
        gal_image = galsim.ImageF(npix_x, npix_y, scale=scale)
        gal_image.setOrigin(0, 0)
        gal_image_list.append(gal_image)
        gal_input_list.append([])
    for i in range(ngal):
        # boundary
        ix = i % ngalx
        iy = i // ngalx
        # each galaxy
        irot = i % nrot_per_gal
        igal = i // nrot_per_gal
        if irot == 0:
            # prepare the base galaxy
            del gal0
            if sim_method == "fft":
                gal0 = _generate_gal_fft(
                    cat_input[igal],
                    mag_zero,
                    rng,
                    gsparams,
                )
            elif sim_method == "mc":
                gal0 = _generate_gal_mc(
                    cat_input[igal],
                    mag_zero,
                    rng,
                    gsparams,
                    npoints=30,
                )
            else:
                raise ValueError("Does not support sim_method=%s" % sim_method)
        else:
            # rotate the base galaxy
            assert gal0 is not None
            ang = np.pi / nrot_per_gal * galsim.radians
            # update gal0
            gal0 = gal0.rotate(ang)

        s01 = rng.np.uniform(low=-0.5, high=0.5) * scale
        s02 = rng.np.uniform(low=-0.5, high=0.5) * scale
        for ii, rr in enumerate(rot_field):
            # base rotation and shear distortion
            gal = gal0.rotate(rr * galsim.radians).shear(g1=g1, g2=g2)
            gal = galsim.Convolve([psf_obj, gal], gsparams=gsparams)
            b = galsim.BoundsI(
                ix * ngrid + buff,
                (ix + 1) * ngrid - 1 + buff,
                iy * ngrid + buff,
                (iy + 1) * ngrid - 1 + buff,
            )
            sub_img = gal_image_list[ii][b]
            # shift with a random offset
            s1, s2 = coord_rotate(s01, s02, xref=0, yref=0, theta=rr)
            if do_shift:
                gal = gal.shift(s1, s2)
            # shift to (ngrid//2, ngrid//2)
            # since I set it as the default center of grids in the simulation
            gal = gal.shift(0.5 * scale, 0.5 * scale)
            gal.drawImage(sub_img, add_to_image=True, method=draw_method)
    gc.collect()
    exposures = [img.array for img in gal_image_list]
    return exposures


def read_cosmos_catalog(filename):
    """Read a COSMOS galaxy catalogue from a FITS file.

    Args:
        filename (str): Path to the FITS catalogue file.

    Returns:
        ndarray: Structured array of catalogue rows.
    """
    cat_input = fitsio.read(filename)
    return cat_input


class CosmosCatalog(object):
    """Filtered COSMOS galaxy catalogue for simulation input.

    Reads a COSMOS FITS catalogue and applies magnitude, half-light
    radius, and morphology-type filters.

    Args:
        filename (str): Path to the COSMOS FITS catalogue.
        max_mag (float | None): Upper magnitude cut (exclusive).
        min_mag (float | None): Lower magnitude cut (inclusive).
        max_hlr (float | None): Maximum half-light radius in arcseconds.
        min_hlr (float | None): Minimum half-light radius in arcseconds.
        gal_type (str): Galaxy morphology filter — ``"mixed"`` (all),
            ``"sersic"`` (single-component), ``"bulgedisk"``
            (two-component), or ``"debug"`` (forced exponential).

    Attributes:
        cat_input (ndarray): Filtered catalogue array.
        ntrain (int): Number of galaxies after filtering.
    """

    def __init__(
        self,
        filename=None,
        max_mag=None,
        min_mag=None,
        max_hlr=None,
        min_hlr=None,
        gal_type="mixed",
    ):
        src = read_cosmos_catalog(filename)
        # Initializing selector mask with all Trues
        sel = np.ones(len(src), dtype=bool)
        # Filtering conditions
        if max_mag is not None:
            sel &= src["mag_auto"] < max_mag
        if min_mag is not None:
            sel &= src["mag_auto"] >= min_mag
        if max_hlr is not None:
            if gal_type == "mixed":
                sel &= (src["hlr"][:, 0:3] < max_hlr).all(axis=1)
            elif gal_type == "sersic":
                sel &= src["hlr"][:, 0] < max_hlr
            elif gal_type == "bulgedisk":
                sel &= (src["hlr"][:, 1:3] < max_hlr).all(axis=1)
        if min_hlr is not None:
            if gal_type == "mixed":
                sel &= (src["hlr"][:, 0:3] >= max_hlr).all(axis=1)
            elif gal_type == "sersic":
                sel &= src["hlr"][:, 0] >= min_hlr
            elif gal_type == "bulgedisk":
                sel &= (src["hlr"][:, 1:3] >= max_hlr).all(axis=1)
        # Applying selector mask
        src = src[sel]
        if gal_type == "mixed":
            pass
        elif gal_type == "sersic":
            src = src[src["use_bulgefit"] == 0]
        elif gal_type == "bulgedisk":
            src = src[src["use_bulgefit"] != 0]
        elif gal_type == "debug":
            # This is used for debug
            src = src[src["use_bulgefit"] == 0]
            # src["sersicfit"][:, 3] = 1.0  # round galaxies
            # src["sersicfit"][:, 2] = 0.5  # only Gaussian
            src["sersicfit"][:, 2] = 1.0  # only Exponential
        else:
            raise ValueError("Do not support gal_type=%s" % gal_type)
        self.cat_input = src
        self.ntrain = len(src)
        return

    def make_catalog(self, rng, n):
        """Draw *n* galaxies at random from the filtered catalogue.

        Args:
            rng (galsim.BaseDeviate): GalSim random number generator.
            n (int): Number of galaxies to sample.

        Returns:
            ndarray: Randomly sampled catalogue rows.

        Raises:
            ValueError: If the catalogue has fewer than *n* entries.
        """
        if self.ntrain < n:
            raise ValueError(
                "There is not enough number of galaxies",
            )
        # nrot_per_gal is the number of rotated galaxies in each subfield
        inds = rng.np.integers(low=0, high=self.ntrain, size=n)
        src = self.cat_input[inds]
        return src


def make_isolated_sim(
    ny,
    nx,
    psf_obj,
    gname,
    seed,
    cat_name=None,
    scale=0.168,
    mag_zero=27.0,
    rot_field=None,
    shear_value=0.02,
    ngrid=64,
    nrot_per_gal=4,
    max_mag=None,
    min_mag=None,
    max_hlr=None,
    min_hlr=None,
    gal_type="mixed",
    do_shift=False,
    npoints=30,
    sim_method="fft",
    buff=0,
    draw_method="auto",
    return_catalog=False,
    verbose=False,
):
    """Make an **isolated** galaxy image simulation on a regular grid.

    Galaxies are drawn from a COSMOS catalogue, placed on an
    ``(ny // ngrid) x (nx // ngrid)`` grid, sheared, convolved with a
    PSF, and rendered into pixel arrays.

    Args:
        ny (int): Number of pixels in the y-direction.
        nx (int): Number of pixels in the x-direction.
        psf_obj (galsim.GSObject): PSF object for convolution.
        gname (str): Shear specification string, e.g. ``"g1-0"`` or
            ``"g2-1"``.  Format is ``"<component>-<index>"`` where
            *index* selects from ``[-shear_value, shear_value, 0]``.
        seed (int): Random seed for the simulation.
        cat_name (str | None): Path to a COSMOS FITS catalogue.
            Defaults to the bundled ``src_cosmos.fits``.
        scale (float): Pixel scale in arcseconds.
        mag_zero (float): Magnitude zero-point (27 for HSC).
        rot_field (list[float] | None): Field rotation angles in
            radians.  Defaults to ``[0.0]``.
        shear_value (float): Amplitude of the applied shear.
        ngrid (int): Stamp size in pixels per galaxy.
        nrot_per_gal (int): Number of shape-noise-cancelling rotations.
        max_mag (float | None): Upper magnitude cut.
        min_mag (float | None): Lower magnitude cut.
        max_hlr (float | None): Maximum half-light radius in arcseconds.
        min_hlr (float | None): Minimum half-light radius in arcseconds.
        gal_type (str): Galaxy morphology filter (``"mixed"``,
            ``"sersic"``, ``"bulgedisk"``, or ``"debug"``).
        do_shift (bool): Whether to apply random sub-pixel offsets.
        npoints (int): Number of random knots (``"mc"`` method only).
        sim_method (str): Galaxy rendering method (``"fft"`` or
            ``"mc"``).
        buff (int): Zero-padding buffer in pixels.
        draw_method (str): GalSim drawing method.
        return_catalog (bool): If ``True``, also return the input
            catalogue.
        verbose (bool): Whether to show log info.

    Returns:
        list[NDArray] | tuple[list[NDArray], ndarray]: Exposure arrays,
        one per field rotation.  If *return_catalog* is ``True``, returns
        ``(exposures, cat_input)``.
    """

    if nx % ngrid != 0:
        raise ValueError("nx is not divisible by ngrid")
    if ny % ngrid != 0:
        raise ValueError("ny is not divisible by ngrid")
    # Basic parameters
    ngalx = int(nx // ngrid)
    ngaly = int(ny // ngrid)
    ngal = ngalx * ngaly
    rng = galsim.BaseDeviate(seed)
    ngeff = max(ngal // nrot_per_gal, 1)
    if cat_name is None:
        cat_name = os.path.join(_data_dir, "src_cosmos.fits")
    cosmos_cat = CosmosCatalog(
        filename=cat_name,
        max_mag=max_mag,
        min_mag=min_mag,
        max_hlr=max_hlr,
        min_hlr=min_hlr,
        gal_type=gal_type,
    )
    cat_input = cosmos_cat.make_catalog(rng=rng, n=ngeff)
    # Get the shear information
    shear_list = np.array([-shear_value, shear_value, 0.0])
    dis_version = int(eval(gname.split("-")[-1]))
    assert dis_version < 3 and dis_version >= 0, "gname is not supported"
    shear_const = shear_list[dis_version]
    g_version = gname.split("-")[0]
    if g_version == "g1":
        g1 = shear_const
        g2 = 0.0
    elif g_version == "g2":
        g1 = 0.0
        g2 = shear_const
    elif g_version == "g1_g2":
        g1 = shear_const
        g2 = 0.02
    elif g_version == "g2_g1":
        g1 = 0.02
        g2 = shear_const
    else:
        raise ValueError("cannot decide g1 or g2")

    if rot_field is None:
        rot_field = [0.0]
    exposures = make_exposure_stamp(
        sim_method=sim_method,
        rng=rng,
        mag_zero=mag_zero,
        psf_obj=psf_obj,
        scale=scale,
        cat_input=cat_input,
        ngalx=ngalx,
        ngaly=ngaly,
        ngrid=ngrid,
        rot_field=rot_field,
        g1=g1,
        g2=g2,
        nrot_per_gal=nrot_per_gal,
        do_shift=do_shift,
        buff=buff,
        draw_method=draw_method,
    )
    if not return_catalog:
        return exposures
    else:
        return exposures, cat_input
