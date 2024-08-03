"""
Simulation of an uncontrolled spacecraft in LEO.
"""

import os
import time
from datetime import datetime
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from Basilisk import __path__

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

from Basilisk.simulation import spacecraft
from Basilisk.utilities import SimulationBaseClass, macros,\
    orbitalMotion, simIncludeGravBody, unitTestSupport, vizSupport
from Basilisk.simulation import simSynch
from Basilisk.architecture import messaging


TIME_STEP_S = 0.05
ACCEL_FACTOR = 1.0
START_TIME = datetime(year=2012, month=5, day=1, hour=0, minute=28, second=30)
NUM_ORBITS = 0.01


if __name__ == "__main__":
    # Spin off visualization process
    vizard = subprocess.Popen(["vizard/Vizard.x86_64", "--args", "-directComm", "tcp://localhost:5556"], stdout=subprocess.DEVNULL)

    # Sim setup
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess(simProcessName)
    simulationTimeStep = macros.sec2nano(TIME_STEP_S)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bskSat"
    scSim.AddModelToTask(simTaskName, scObject)

    # Gravity setup
    gravFactory = simIncludeGravBody.gravBodyFactory()
    grav_bodies = gravFactory.createBodies("earth", "moon", "sun")
    grav_bodies["earth"].isCentralBody = True
    grav_bodies["earth"].useSphericalHarmonicsGravityModel(bskPath + "/supportData/LocalGravData/GGM03S.txt", 10)
    mu_earth = grav_bodies["earth"].mu
    mu = mu_earth
    gravFactory.addBodiesTo(scObject)
    spice_time = START_TIME.strftime("%Y %b %d %X TDB")
    gravFactory.createSpiceInterface(bskPath + "/supportData/EphemerisData/",
        time=spice_time, epochInMsg=True)
    gravFactory.spiceObject.zeroBase = "Earth"
    planetStateOutMsgs = {
        "earth": gravFactory.spiceObject.planetStateOutMsgs[0],
        "moon": gravFactory.spiceObject.planetStateOutMsgs[1],
        "sun": gravFactory.spiceObject.planetStateOutMsgs[2],
    }
    scSim.AddModelToTask(simTaskName, gravFactory.spiceObject)

    # Initial conditions
    oe = orbitalMotion.ClassicElements()
    oe.a = 7e6  # meters
    oe.e = 0.0001
    oe.i = 33.3 * macros.D2R
    oe.Omega = 48.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 85.3 * macros.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    oe = orbitalMotion.rv2elem(mu, rN, vN)
    scObject.hub.r_CN_NInit = rN  # meters
    scObject.hub.v_CN_NInit = vN  # meters per second
    n = np.sqrt(mu / oe.a / oe.a / oe.a)
    P = 2. * np.pi / n
    simulationTime = macros.sec2nano(P * NUM_ORBITS)

    # Logging
    numDataPoints = 400
    samplingTime = unitTestSupport.samplingTime(simulationTime,
                                                simulationTimeStep,
                                                numDataPoints)
    dataLog = scObject.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, dataLog)

    # Final setup
    clockSync = simSynch.ClockSynch()
    clockSync.accelFactor = ACCEL_FACTOR
    scSim.AddModelToTask(simTaskName, clockSync)
    vizSupport.enableUnityVisualization(scSim, simTaskName, scObject,
                                        liveStream=True)
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    scSim.SetProgressBar(True)

    # Run the simulation
    t0 = time.time()
    scSim.ExecuteSimulation()
    print(f"Simulation complete after {time.time() - t0:0.1f} seconds.")
    vizard.kill()

    # Plot results
    posData = dataLog.r_BN_N
    velData = dataLog.v_BN_N
    np.set_printoptions(precision=16)
    plt.close("all")  # clears out plots from earlier test runs
    plt.figure(1)
    fig = plt.gcf()
    ax = fig.gca()
    ax.ticklabel_format(useOffset=False, style='plain')
    for idx in range(3):
        plt.plot(dataLog.times() * macros.NANO2SEC / P, posData[:, idx] / 1000.,
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='$r_{BN,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [orbits]')
    plt.ylabel('Inertial Position [km]')
    figureList = {}
    pltName = fileName + "1" + "LEO"
    figureList[pltName] = plt.figure(1)
    plt.figure(2)
    fig = plt.gcf()
    ax = fig.gca()
    ax.ticklabel_format(useOffset=False, style='plain')
    smaData = []
    for idx in range(0, len(posData)):
        oeData = orbitalMotion.rv2elem(mu, posData[idx], velData[idx])
        smaData.append(oeData.a / 1000.)
    plt.plot(posData[:, 0] * macros.NANO2SEC / P, smaData, color='#aa0000')
    plt.xlabel('Time [orbits]')
    plt.ylabel('SMA [km]')
    pltName = fileName + "2" + "LEO"
    figureList[pltName] = plt.figure(2)
    plt.show()
    plt.close("all")
