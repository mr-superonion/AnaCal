import os
import gc
import galsim
import logging
import numpy as np
import astropy.io.fits as pyfits

from . import dtype

img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)
nrot_default = 4
ngrid_stamp = 64


def coord_rotate(x, y, xref, yref, theta):
    """Rotates coordinates by an angle theta (anticlockwise)

    Args:
        x (ndarray):    input coordinates [x]
        y (ndarray):    input coordinates [y]
        xref (float):   reference point [x]
        yref (float):   reference point [y]
        theta (float):  rotation angle [rads]
    Returns:
        x2 (ndarray):   rotated coordiantes [x]
        y2 (ndarray):   rotated coordiantes [y]
    """
    xu = x - xref
    yu = y - yref
    x2 = np.cos(theta) * xu - np.sin(theta) * yu + xref
    y2 = np.sin(theta) * xu + np.cos(theta) * yu + yref
    return x2, y2


def generate_cosmos_gal(record, truncr=5.0, gsparams=None):
    """Generates COSMOS galaxies; modified version of
    https://github.com/GalSim-developers/GalSim/blob/releases/2.3/galsim/scene.py#L626

    Args:
        record (ndarray):   one row of the COSMOS galaxy catalog
        truncr (float):     truncation ratio
        gsparams:           An GSParams argument.
    Returns:
        gal:    Galsim galaxy
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
    def _galsim_round_sersic(n, sersic_prec):
        return float(int(n / sersic_prec + 0.5)) * sersic_prec

    bparams = record["bulgefit"]
    sparams = record["sersicfit"]
    use_bulgefit = record["use_bulgefit"]
    if use_bulgefit:
        # Bulge parameters:
        # Minor-to-major axis ratio:
        bulge_hlr = record["hlr"][1]
        bulge_flux = record["flux"][1]
        disk_hlr = record["hlr"][2]
        disk_flux = record["flux"][2]
        if truncr <= 0.99:
            btrunc = None
            bulge = galsim.DeVaucouleurs(
                flux=bulge_flux, half_light_radius=bulge_hlr, gsparams=gsparams
            )
            disk = galsim.Exponential(
                flux=disk_flux, half_light_radius=disk_hlr, gsparams=gsparams
            )
        else:
            btrunc = bulge_hlr * truncr
            bulge = galsim.DeVaucouleurs(
                flux=bulge_flux,
                half_light_radius=bulge_hlr,
                trunc=btrunc,
                gsparams=gsparams,
            )
            dtrunc = disk_hlr * truncr
            disk = galsim.Sersic(
                1.0,
                flux=disk_flux,
                half_light_radius=disk_hlr,
                trunc=dtrunc,
                gsparams=gsparams,
            )
        # Apply shears for intrinsic shape.
        bulge_q = bparams[11]
        bulge_beta = bparams[15] * galsim.radians
        if bulge_q < 1.0:  # pragma: no branch
            bulge = bulge.shear(q=bulge_q, beta=bulge_beta)
        disk_q = bparams[3]
        disk_beta = bparams[7] * galsim.radians
        if disk_q < 1.0:  # pragma: no branch
            disk = disk.shear(q=disk_q, beta=disk_beta)
        # Then combine the two components of the galaxy.
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

        gal_q = sparams[3]
        gal_beta = sparams[7] * galsim.radians
        if truncr <= 0.99:
            btrunc = None
            gal = galsim.Sersic(
                gal_n, flux=gal_flux, half_light_radius=gal_hlr, gsparams=gsparams
            )
        else:
            btrunc = gal_hlr * truncr
            gal = galsim.Sersic(
                gal_n,
                flux=gal_flux,
                half_light_radius=gal_hlr,
                trunc=btrunc,
                gsparams=gsparams,
            )
        # Apply shears for intrinsic shape.
        if gal_q < 1.0:  # pragma: no branch
            gal = gal.shear(q=gal_q, beta=gal_beta)

    return gal


def _basic_gals(
    seed,
    gal_image,
    magzero,
    psf_obj,
    scale,
    bigfft,
    cat_input,
    ngalx,
    ngaly,
    ngrid,
    rot2,
    g1,
    g2,
    nrot,
    shifts,
):
    ngal = ngalx * ngaly
    logging.info("Making Basic Simulation. ID: %d" % (seed))
    gal0 = None
    for i in range(ngal):
        # boundary
        ix = i % ngalx
        iy = i // ngalx
        b = galsim.BoundsI(
            ix * ngrid,
            (ix + 1) * ngrid - 1,
            iy * ngrid,
            (iy + 1) * ngrid - 1,
        )
        # each galaxy
        irot = i % nrot
        if irot == 0:
            del gal0
            ig = i // nrot
            ss = cat_input[ig]
            # gal0  =  cosmos_cat.makeGalaxy(gal_type='parametric',\
            #             index=ss['index'],gsparams=bigfft)
            gal0 = generate_cosmos_gal(ss, truncr=-1.0, gsparams=bigfft)
            # accounting for zeropoint difference between COSMOS HST and HSC
            # HSC's i-band coadds zero point is 27
            flux = 10 ** ((magzero - ss["mag_auto"]) / 2.5)
            # flux_scaling=   2.587
            gal0 = gal0.withFlux(flux)
            # rescale the radius by 'rescale' and keep surface brightness the
            # same
            rescale = np.random.uniform(0.95, 1.05)
            gal0 = gal0.expand(rescale)
            # rotate by 'ang'
            ang = (np.random.uniform(0.0, np.pi * 2.0) + rot2) * galsim.radians
            gal0 = gal0.rotate(ang)
        else:
            assert gal0 is not None
            ang = np.pi / nrot * galsim.radians
            # update gal0
            gal0 = gal0.rotate(ang)
        # shear distortion
        gal = gal0.shear(g1=g1, g2=g2)
        # shift to (ngrid//2,ngrid//2)
        # the random shift is relative to this point
        gal = galsim.Convolve([psf_obj, gal], gsparams=bigfft)
        gal = gal.shift(0.5 * scale, 0.5 * scale)
        if shifts is not None:
            gal = gal.shift(shifts[i, 0], shifts[i, 1])
        # draw galaxy
        sub_img = gal_image[b]
        gal.drawImage(sub_img, add_to_image=True)
        del gal, b, sub_img
    gc.collect()
    return


def _random_gals(
    seed,
    gal_image,
    magzero,
    psf_obj,
    scale,
    bigfft,
    cat_input,
    ngalx,
    ngaly,
    ngrid,
    rot2,
    g1,
    g2,
    nrot,
    shifts,
    npoints=30,
):
    ud = galsim.UniformDeviate(seed)
    # use galaxies with random knots
    # we only support three versions of small galaxies
    logging.info("Making galaxies with Random Knots.")
    gal0 = None

    for iy in range(ngaly):
        for ix in range(ngalx):
            b = galsim.BoundsI(
                ix * ngrid,
                (ix + 1) * ngrid - 1,
                iy * ngrid,
                (iy + 1) * ngrid - 1,
            )
            sub_img = gal_image[b]
            #
            ii = iy * ngalx + ix
            irot = ii % nrot
            if irot == 0:
                del gal0
                ig = ii // nrot
                ss = cat_input[ig]
                flux = 10 ** ((magzero - ss["mag_auto"]) / 2.5)
                gal0 = galsim.RandomKnots(
                    half_light_radius=ss["flux_radius"] * 0.05,
                    npoints=npoints,
                    flux=flux,
                    rng=ud,
                    gsparams=bigfft,
                )
                ang = (np.random.uniform(0.0, np.pi * 2.0) + rot2) * galsim.radians
            else:
                assert gal0 is not None
                ang = np.pi / nrot * galsim.radians
                gal0 = gal0.rotate(ang)
            # Shear the galaxy
            gal = gal0.shear(g1=g1, g2=g2)
            gal = galsim.Convolve([psf_obj, gal], gsparams=bigfft)
            gal = gal.shift(0.5 * scale, 0.5 * scale)
            if shifts is not None:
                gal = gal.shift(shifts[ii, 0], shifts[ii, 1])
            # Draw the galaxy image
            gal.drawImage(sub_img, add_to_image=True)
            del gal, b, sub_img
    gc.collect()
    return


def make_isolate_sim(
    gal_type,
    ny,
    nx,
    psf_obj,
    gname,
    seed,
    catname=None,
    scale=0.168,
    magzero=27.0,
    rot2=0,
    shear_value=0.02,
    ngrid=64,
    nrot=nrot_default,
    mag_cut=None,
    do_shift=None,
    npoints=30,
):
    """Makes basic **isolated** galaxy image simulation.

    Args:
        gal_type (str):         galaxy tpye ("random" or "basic")
        ny (int):               number of pixels in y direction
        nx (int):               number of pixels in y direction
        psf_obj (PSF):          input PSF object of galsim
        gname (str):            shear distortion setup
        seed (int):             index of the simulation
        catname (str):          input catalog name
        scale (float):          pixel scale
        magzero (float):        magnitude zero point [27 for HSC]
        rot2 (float):           additional rotation angle
        shear_value (float):    shear distortion amplitude
        ngrid (int):            stampe size
        nrot (int):             number of rotations
        mag_cut (float):        magnitude cut of the input catalog
        do_shift (bool):        whether do shfits
    """
    np.random.seed(seed)

    if nx % ngrid != 0:
        raise ValueError("nx is not divisible by ngrid")
    if ny % ngrid != 0:
        raise ValueError("ny is not divisible by ngrid")
    # Basic parameters
    ngalx = int(nx // ngrid)
    ngaly = int(ny // ngrid)
    ngal = ngalx * ngaly

    if catname is None:
        catname = os.path.join(img_dir, "cat_used.fits")
    cat_input = pyfits.getdata(catname)
    if mag_cut is not None:
        cat_input = cat_input[cat_input["mag_auto"] < mag_cut]
    ntrain = len(cat_input)
    if not ntrain > ngal:
        raise ValueError("mag_cut is too small")
    ngeff = max(ngal // nrot, 1)
    inds = np.random.randint(0, ntrain, ngeff)
    cat_input = cat_input[inds]

    # Get the shear information
    shear_list = np.array([-shear_value, 0.0, shear_value])
    shear_list = shear_list[[eval(i) for i in gname.split("-")[-1]]]
    if gname.split("-")[0] == "g1":
        g1 = shear_list[0]
        g2 = 0.0
    elif gname.split("-")[0] == "g2":
        g1 = 0.0
        g2 = shear_list[0]
    else:
        raise ValueError("cannot decide g1 or g2")
    logging.info(
        "Processing for %s, and shears for four redshift bins are %s."
        % (gname, shear_list)
    )

    gal_image = galsim.ImageF(nx, ny, scale=scale)
    gal_image.setOrigin(0, 0)
    bigfft = galsim.GSParams(maximum_fft_size=10240)
    if do_shift:
        shifts = np.random.uniform(low=-0.5, high=0.5, size=(ngal, 2)) * scale
    else:
        shifts = None
    if gal_type == "basic":
        _basic_gals(
            seed,
            gal_image,
            magzero,
            psf_obj,
            scale,
            bigfft,
            cat_input,
            ngalx,
            ngaly,
            ngrid,
            rot2,
            g1,
            g2,
            nrot,
            shifts,
        )
    elif gal_type == "random":
        _random_gals(
            seed,
            gal_image,
            magzero,
            psf_obj,
            scale,
            bigfft,
            cat_input,
            ngalx,
            ngaly,
            ngrid,
            rot2,
            g1,
            g2,
            nrot,
            shifts,
            npoints,
        )
    else:
        raise ValueError("gal_type should cotain 'basic' or 'random'!!")
    psf_array = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid_stamp, ny=ngrid_stamp, scale=scale)
        .array
    )
    nstd_f = 0.0
    noise_pow = np.ones((ngrid, ngrid)) * nstd_f**2.0 * ngrid**2.0

    return dtype.ImageData(
        image=gal_image.array,
        psf=psf_array,
        scale=scale,
        mag_zero=magzero,
        noise_pow=noise_pow,
    )


def make_noise_sim(
    out_dir,
    infname,
    ind0,
    ny=6400,
    nx=6400,
    scale=0.168,
    do_write=True,
    return_array=False,
):
    """Makes pure noise for galaxy image simulation.

    Args:
        out_dir (str):          output directory
        ind0 (int):             index of the simulation
        ny (int):               number of pixels in y direction
        nx (int):               number of pixels in x direction
        do_write (bool):        whether write output [default: True]
        return_array (bool):    whether return galaxy array [default: False]
    """
    logging.info("begining for field %04d" % (ind0))
    out_fname = os.path.join(out_dir, "noi%04d.fits" % (ind0))
    if os.path.exists(out_fname):
        if do_write:
            logging.info("Nothing to write.")
        if return_array:
            return pyfits.getdata(out_fname)
        else:
            return None
    logging.info("simulating noise for field %s" % (ind0))
    variance = 0.01
    ud = galsim.UniformDeviate(ind0 * 10000 + 1)

    # setup the galaxy image and the noise image
    noi_image = galsim.ImageF(nx, ny, scale=scale)
    noi_image.setOrigin(0, 0)
    noise_obj = galsim.getCOSMOSNoise(
        file_name=infname, rng=ud, cosmos_scale=scale, variance=variance
    )
    noise_obj.applyTo(noi_image)
    if do_write:
        pyfits.writeto(out_fname, noi_image.array)
    if return_array:
        return noi_image.array
    return
