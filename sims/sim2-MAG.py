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

from Basilisk.simulation import spacecraft, simSynch, coarseSunSensor,\
    eclipse, magnetometer, magneticFieldWMM
from Basilisk.utilities import SimulationBaseClass, macros,\
    orbitalMotion, simIncludeGravBody, unitTestSupport, vizSupport,\
    simSetPlanetEnvironment
from Basilisk.simulation import simSynch
from Basilisk.architecture import messaging
from Basilisk.topLevelModules import pyswice
from Basilisk.utilities.pyswice_spk_utilities import spkRead


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

    # Magnetic field model
    magModule = magneticFieldWMM.MagneticFieldWMM()
    magModule.ModelTag = "WMM"
    magModule.dataPath = bskPath + "/supportData/MagneticField/"
    magModule.planetPosInMsg.subscribeTo(planetStateOutMsgs["earth"])
    epochMsg = unitTestSupport.timeStringToGregorianUTCMsg(
        START_TIME.strftime("%Y %b %d, %X (UTC)"))
    magModule.epochInMsg.subscribeTo(epochMsg)
    magModule.addSpacecraftToModel(scObject.scStateOutMsg)
    scSim.AddModelToTask(simTaskName, magModule)

    # Spacecraft setup
    I = [900., 0., 0.,
         0., 800., 0.,
         0., 0., 600.]
    scObject.hub.mHub = 750.0  # spacecraft mass (kg)
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # position vector of body-fixed point B relative to CM (m)
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

    # Initial orbit conditions
    oe = orbitalMotion.ClassicElements()
    oe.a = 700 * 1e3 + 6.371e6  # meters
    oe.e = 0.0001
    oe.i = 33.3 * macros.D2R
    oe.Omega = 48.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 85.3 * macros.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    oe = orbitalMotion.rv2elem(mu, rN, vN)
    n = np.sqrt(mu / oe.a / oe.a / oe.a)
    P = 2. * np.pi / n
    simulationTime = macros.sec2nano(P * NUM_ORBITS)

    # Initial spacecraft states/rates
    scObject.hub.r_CN_NInit = rN  # position vector (m)
    scObject.hub.v_CN_NInit = vN  # velocity vector (m/s)
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]  # attitude MRP
    scObject.hub.omega_BN_BInit = [[1.*macros.D2R],
                                   [2.*macros.D2R],
                                   [3.*macros.D2R]]  # angular rates (rad/s)

    # Eclipse model
    eclipseObject = eclipse.Eclipse()
    eclipseObject.addSpacecraftToModel(scObject.scStateOutMsg)
    eclipseObject.addPlanetToModel(planetStateOutMsgs["earth"])
    eclipseObject.sunInMsg.subscribeTo(planetStateOutMsgs["sun"])
    scSim.AddModelToTask(simTaskName, eclipseObject)

    # Define CSS units. Parameters common to all are hardcoded, while parameters
    # that differ between units are given as parameters.
    def setupCSS(CSS, tag, pos, direction):
        CSS.fov = 90. * macros.D2R
        CSS.scaleFactor = 1.0
        CSS.maxOutput = 1.0
        CSS.minOutput = 0
        CSS.kellyFactor = 0.1
        CSS.senBias = 0.0  # normalized sensor bias
        CSS.senNoiseStd = 0.01  # normalized sensor noise
        CSS.sunInMsg.subscribeTo(planetStateOutMsgs["sun"])
        CSS.stateInMsg.subscribeTo(scObject.scStateOutMsg)
        CSS.sunEclipseInMsg.subscribeTo(eclipseObject.eclipseOutMsgs[0])
        CSS.ModelTag = tag
        CSS.r_B = np.array(pos)
        CSS.nHat_B = np.array(direction)

    CSS0 = coarseSunSensor.CoarseSunSensor()
    setupCSS(CSS0, "CSS0_sensor", [1, 0, 0], [1, 0, 0])
    CSS1 = coarseSunSensor.CoarseSunSensor()
    setupCSS(CSS1, "CSS1_sensor", [-1, 0, 0], [-1, 0, 0])
    CSS2 = coarseSunSensor.CoarseSunSensor()
    setupCSS(CSS2, "CSS2_sensor", [0, 1, 0], [0, 1, 0])
    CSS3 = coarseSunSensor.CoarseSunSensor()
    setupCSS(CSS3, "CSS3_sensor", [0, -1, 0], [0, -1, 0])
    CSS4 = coarseSunSensor.CoarseSunSensor()
    setupCSS(CSS4, "CSS4_sensor", [0, 0, 1], [0, 0, 1])
    CSS5 = coarseSunSensor.CoarseSunSensor()
    setupCSS(CSS5, "CSS5_sensor", [0, 0, -1], [0, 0, -1])
    cssList = [CSS0, CSS1, CSS2, CSS3, CSS4, CSS5]

    # Add CSS units to sim
    for css in cssList:
        scSim.AddModelToTask(simTaskName, css)

    # Three-Axis Magnetometer (TAM)
    TAM = magnetometer.Magnetometer()
    TAM.ModelTag = "TAM_sensor"
    TAM.scaleFactor = 1.0
    #TAM.senNoiseStd = [100e-9, 100e-9, 100e-9]
    TAM.stateInMsg.subscribeTo(scObject.scStateOutMsg)
    scSim.AddModelToTask(simTaskName, TAM)

    # Logging
    numDataPoints = 400
    samplingTime = unitTestSupport.samplingTime(simulationTime,
                                                simulationTimeStep,
                                                numDataPoints)
    dataLog = scObject.scStateOutMsg.recorder(samplingTime)
    cssLogs = [css.cssDataOutMsg.recorder(samplingTime) for css in cssList]
    tamLog = TAM.tamDataOutMsg.recorder(samplingTime)
    magLog = magModule.envOutMsgs[0].recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, dataLog)
    for cssl in cssLogs:
        scSim.AddModelToTask(simTaskName, cssl)
    scSim.AddModelToTask(simTaskName, tamLog)
    scSim.AddModelToTask(simTaskName, magLog)
    TAM.magInMsg.subscribeTo(magModule.envOutMsgs[0])

    # Final setup
    clockSync = simSynch.ClockSynch()
    clockSync.accelFactor = ACCEL_FACTOR
    scSim.AddModelToTask(simTaskName, clockSync)
    viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject,
                                              liveStream=True,
                                              cssList=[cssList])
    print("This is where vizard prints a 1:")
    vizSupport.setInstrumentGuiSetting(viz, viewCSSPanel=True,
                                       viewCSSCoverage=True,
                                       viewCSSBoresight=True,
                                       showCSSLabels=True)
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    scSim.SetProgressBar(True)

    # Run the simulation
    t0 = time.time()
    scSim.ExecuteSimulation()
    print(f"Simulation complete after {time.time() - t0:0.1f} seconds.")
    vizard.kill()

    # Plot results
    t = dataLog.times() * macros.NANO2SEC
    posData = dataLog.r_BN_N
    velData = dataLog.v_BN_N
    cssData = [cssl.OutputData for cssl in cssLogs]
    tamData = tamLog.tam_S
    magData = magLog.magField_N
    np.set_printoptions(precision=16)
    plt.close("all")  # clears out plots from earlier test runs

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.ticklabel_format(useOffset=False, style='plain')
    for idx in range(3):
        ax1.plot(t, posData[:, idx] / 1000.,
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='$r_{BN,' + str(idx) + '}$')
    ax1.legend(loc='lower right')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Inertial Position (km)')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.ticklabel_format(useOffset=False, style='plain')
    smaData = []
    for idx in range(0, len(posData)):
        oeData = orbitalMotion.rv2elem(mu, posData[idx], velData[idx])
        smaData.append(oeData.a / 1000.)
    ax2.plot(t, smaData, color='#aa0000')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('SMA (km)')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.ticklabel_format(useOffset=False, style='plain')
    for i, cssl in enumerate(cssLogs):
        ax3.plot(t, cssl.OutputData, label=f"CSS{i}")
    ax3.legend()
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('CSS readings (mA)')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.ticklabel_format(useOffset=False, style='sci')
    ax4.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    for idx in range(3):
        ax4.plot(t, tamData[:, idx] * 1e9,
                 color=unitTestSupport.getLineColor(idx, 3), 
                 label=r'$TAM_{' + str(idx) + '}$')
    ax4.legend(loc='lower right')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Magnetic Field (nT)')

    plt.show()
    plt.close("all")
