"""
Simulation of an uncontrolled spacecraft in LEO.
"""

import os
import time
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
    grav_factory = simIncludeGravBody.gravBodyFactory()
    grav_bodies = grav_factory.createBodies("earth", "moon", "sun")
    grav_bodies["earth"].isCentralBody = True
    grav_bodies["earth"].useSphericalHarmonicsGravityModel(bskPath + "/supportData/LocalGravData/GGM03S.txt", 10)
    mu_earth = grav_bodies["earth"].mu
    mu = mu_earth
    grav_factory.addBodiesTo(scObject)
    spice_object = grav_factory.createSpiceInterface(time="2012 MAY 1 00:28:30.0 TDB", epochInMsg=True)
    spice_object.zeroBase = "Earth"
    spice_object.addPlanetNames(messaging.StringVector(["EARTH", "MOON", "SUN"]))
    spice_object.loadSpiceKernel("de421.bsp", bskPath + "/supportData/EphemerisData/")
    scSim.AddModelToTask(simTaskName, spice_object)

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
    simulationTime = macros.sec2nano(P/100)  # Run for 1/100 of an orbit

    # Logging
    numDataPoints = 400
    samplingTime = unitTestSupport.samplingTime(simulationTime,
                                                simulationTimeStep,
                                                numDataPoints)
    dataLog = scObject.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, dataLog)

    # Final setup
    clockSync = simSynch.ClockSynch()
    clockSync.accelFactor = 1.0
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
