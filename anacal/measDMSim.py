#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 20082014 LSST Corpoalphan.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import os
import galsim
import numpy as np
import astropy.io.fits as pyfits
from lsst.utils import getPackageDir

# for pipe task
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

# lsst.afw...
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.image as afwImg
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
from lsst.meas.deblender import SourceDeblendTask
from lsst.meas.algorithms import SourceDetectionTask
from lsst.pipe.tasks.scaleVariance import ScaleVarianceTask
from lsst.meas.base import SingleFrameMeasurementTask, CatalogCalculationTask


class measDMSimConfig(pexConfig.Config):
    "config"
    rootDir = pexConfig.Field(dtype=str, default="./", doc="Root Diectory")
    expPrefix = pexConfig.Field(
        dtype=str, default="expDir", doc="prefix of input exposure"
    )
    srcPrefix = pexConfig.Field(
        dtype=str, default="outcomeHSM2", doc="prefiex of output src"
    )
    doWrite = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether write outcome",
    )
    doAddFP = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether add footprint",
    )
    doDeblend = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether do deblending",
    )
    doScaleVariance = pexConfig.Field(
        dtype=bool, default=True, doc="Scale variance plane using empirical noise?"
    )
    detection = pexConfig.ConfigurableField(
        target=SourceDetectionTask, doc="Detect sources"
    )
    scaleVariance = pexConfig.ConfigurableField(
        target=ScaleVarianceTask, doc="Variance rescaling"
    )
    deblend = pexConfig.ConfigurableField(
        target=SourceDeblendTask, doc="Split blended source into their components"
    )
    measurement = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask, doc="Measure sources"
    )
    catalogCalculation = pexConfig.ConfigurableField(
        target=CatalogCalculationTask,
        doc="Subtask to run catalogCalculation plugins on catalog",
    )

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        self.detection.isotropicGrow = True
        self.detection.reEstimateBackground = False
        self.detection.thresholdValue = 5.0

        self.deblend.propagateAllPeaks = True
        self.deblend.maxFootprintArea = 250 * 250
        self.deblend.maxFootprintSize = 400

        self.measurement.load(
            os.path.join(getPackageDir("obs_subaru"), "config", "hsm.py")
        )
        self.load(os.path.join(getPackageDir("obs_subaru"), "config", "cmodel.py"))


class measDMSimTask(pipeBase.CmdLineTask):
    ConfigClass = measDMSimConfig
    _DefaultName = "measDMSim"

    def __init__(self, schema, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.schema = schema
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("detection", schema=self.schema)
        self.schema.addField("ipos", type=np.int32, doc="the position of the stamps")
        if self.config.doDeblend:
            self.makeSubtask("deblend", schema=self.schema)
        self.makeSubtask(
            "measurement", schema=self.schema, algMetadata=self.algMetadata
        )
        self.makeSubtask("catalogCalculation", schema=self.schema)
        if self.config.doScaleVariance:
            self.makeSubtask("scaleVariance")
        self.schema.addField(
            "undetected", type=np.int32, doc="wheter galaxy is undetected"
        )

    def measDM(self, prepend):
        # Read galaxy exposure
        expDir = os.path.join(self.config.rootDir, self.config.expPrefix)
        expfname = "image%s.fits" % (prepend)
        expfname = os.path.join(expDir, expfname)
        if not os.path.exists(expfname):
            self.log.info("cannot find the exposure")
            return None
        exposure = afwImg.ExposureF.readFits(expfname)
        exposure.getMask().getArray()[:, :] = 0
        if not exposure.hasPsf():
            self.log.info("exposure doesnot have PSF")
            return None
        # Read sources
        sourceDir = os.path.join(self.config.rootDir, self.config.srcPrefix)
        sourceFname = os.path.join(sourceDir, "src%s.fits" % (prepend))
        if os.path.exists(sourceFname):
            sources = afwTable.SourceCatalog.readFits(sourceFname)
        else:
            sources = self.measureSource(exposure, expfname)
            if sources is None:
                self.log.info("Cannot read sources")
                return None
            if self.config.doWrite:
                sources.writeFits(sourceFname)

        return pipeBase.Struct(exposure=exposure, sources=sources)

    def measureSource(self, exposure, expfname=None):
        table = afwTable.SourceTable.make(self.schema)
        sources = afwTable.SourceCatalog(table)
        table.setMetadata(self.algMetadata)
        detRes = self.detection.run(table=table, exposure=exposure, doSmooth=True)
        sources = detRes.sources
        if self.config.doScaleVariance:
            varScale = self.scaleVariance.run(exposure.maskedImage)
            exposure.getMetadata().add("variance_scale", varScale)
        if self.config.doDeblend:
            # do deblending
            self.deblend.run(exposure=exposure, sources=sources)
        # do measurement
        self.measurement.run(measCat=sources, exposure=exposure)
        # measurement on the catalog level
        self.catalogCalculation.run(sources)
        if expfname is not None:
            exposure.writeFits(expfname)
        return sources
