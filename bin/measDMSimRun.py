#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 20082014 LSST Corporation.
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
# python lib
import os
import numpy as np
from astropy.table import Table
import astropy.io.fits as pyfits

# lsst Tasks
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.image as afwImg
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord

from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

import lsst.obs.subaru.filterFraction
from anacal.measDMSim import measDMSimTask


class runMeasDMConfig(pexConfig.Config):
    "config"
    measDMSim = pexConfig.ConfigurableField(
        target=measDMSimTask, doc="Subtask to run measurement"
    )
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)


class runMeasDMTask(pipeBase.CmdLineTask):
    _DefaultName = "runMeasDM"
    ConfigClass = runMeasDMConfig

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("measDMSim", schema=self.schema)

    @pipeBase.timeMethod
    def runDataRef(self, isim):
        info = Table.read("info.csv")
        ifield = int(info[isim]["num"])
        self.log.info("begining for field %05d" % (ifield))
        inputdir = "expDir/"

        inFname = os.path.join(inputdir, "image-%05d.fits" % (ifield))
        if not os.path.exists(inFname):
            self.log.info("Cannot find the input exposure")
            return
        self.measDMSim.measDM("-%05d" % ifield)
        return

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser"""
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        return parser
    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass
    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass
    def writeMetadata(self, ifield):
        pass
    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass


class runMeasDMDriverConfig(pexConfig.Config):
    perGroup = pexConfig.Field(dtype=int, default=100, doc="data per field")
    runMeasDM = pexConfig.ConfigurableField(
        target=runMeasDMTask, doc="runMeasDM task to run on multiple cores"
    )

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)


class runMeasDMRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minGroup = parsedCmd.minGroup
        maxGroup = parsedCmd.maxGroup
        return [(ref, kwargs) for ref in range(minGroup, maxGroup)]


def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)


class runMeasDMDriverTask(BatchPoolTask):
    ConfigClass = runMeasDMDriverConfig
    RunnerClass = runMeasDMRunner
    _DefaultName = "runMeasDMDriver"

    def __reduce__(self):
        """Pickler"""
        return unpickle, (
            self.__class__,
            [],
            dict(
                config=self.config,
                name=self._name,
                parentTask=self._parentTask,
                log=self.log,
            ),
        )

    def __init__(self, **kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.makeSubtask("runMeasDM")

    @abortOnError
    def runDataRef(self, Id):
        self.log.info("beginning group %d" % (Id))
        perGroup = self.config.perGroup
        fMin = perGroup * Id
        fMax = perGroup * (Id + 1)
        fieldList = range(fMin, fMax)
        # Prepare the pool
        pool = Pool("runMeasDM")
        pool.cacheClear()
        pool.map(self.process, fieldList)
        return

    def process(self, cache, isim):
        self.runMeasDM.runDataRef(isim)
        self.log.info("finish field %05d" % (isim))
        return

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument(
            "--minGroup", type=int, default=0, help="minimum group number"
        )
        parser.add_argument(
            "--maxGroup", type=int, default=10, help="maximum group number"
        )
        return parser

    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass
    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass
    def writeMetadata(self, ifield):
        pass
    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass
    def _getConfigName(self):
        return None
    def _getEupsVersionsName(self):
        return None
    def _getMetadataName(self):
        return None


if __name__ == "__main__":
    runMeasDMDriverTask.parseAndSubmit()
