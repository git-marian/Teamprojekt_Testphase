#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.0),
    on June 16, 2026, at 22:19
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.0'
expName = 'Teamprojekt'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1280, 800]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\maria\\Documents\\Teamprojekt_Testphase\\Teamprojekt_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('keyInstructionContinue') is None:
        # initialise keyInstructionContinue
        keyInstructionContinue = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyInstructionContinue',
        )
    if deviceManager.getDevice('keyElementMotionExplained') is None:
        # initialise keyElementMotionExplained
        keyElementMotionExplained = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyElementMotionExplained',
        )
    if deviceManager.getDevice('keyGroupMotionExplained') is None:
        # initialise keyGroupMotionExplained
        keyGroupMotionExplained = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyGroupMotionExplained',
        )
    if deviceManager.getDevice('keyPractice') is None:
        # initialise keyPractice
        keyPractice = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyPractice',
        )
    if deviceManager.getDevice('keyStart') is None:
        # initialise keyStart
        keyStart = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyStart',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('keyBlockContinue') is None:
        # initialise keyBlockContinue
        keyBlockContinue = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyBlockContinue',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "InstructionScreen" ---
    keyInstructionContinue = keyboard.Keyboard(deviceName='keyInstructionContinue')
    textInstructions = visual.TextStim(win=win, name='textInstructions',
        text='Im folgenden Teil der Studie siehst du wiederholt kurze Abfolgen von drei Kreisen. \n\nZwischen den beiden Darstellungen können die Kreise auf unterschiedliche Weise wahrgenommen werden:\n\n\n• Bei einer Gruppenbewegung scheinen sich alle drei Kreise gemeinsam als Gruppe nach rechts zu bewegen.\n\n• Bei einer Elementbewegung scheint der linke Kreis zu verschwinden, während gleichzeitig rechts ein neuer Kreis erscheint. Die beiden mittleren Kreise scheinen dabei an ihrer Position zu bleiben. \n\nDeine Aufgabe besteht darin, bei jedem Durchgang anzugeben, welche Wahrnehmung du hattest.\n\nDrücke:\n\nF = Elementbewegung\n\nJ = Gruppenbewegung\n\nEs gibt keine richtigen oder falschen Antworten. Bitte gib jeweils die Wahrnehmung an, die deinem Eindruck am besten entspricht.\n\n Zunächst siehst du zwei Beispiele und bearbeitest anschließend einige Übungsdurchgänge.\n\nDrücke die Leertaste, um fortzufahren.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "exampleFrame1" ---
    ternusExampleLeft = visual.ShapeStim(
        win=win, name='ternusExampleLeft',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    ternusExampleMiddle = visual.ShapeStim(
        win=win, name='ternusExampleMiddle',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    ternusExampleRight = visual.ShapeStim(
        win=win, name='ternusExampleRight',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    FixationCrossExample = visual.ShapeStim(
        win=win, name='FixationCrossExample', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "exampleFrame2" ---
    ternusExampleLeft_2 = visual.ShapeStim(
        win=win, name='ternusExampleLeft_2',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    ternusExampleMiddle_2 = visual.ShapeStim(
        win=win, name='ternusExampleMiddle_2',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    ternusExampleRight_2 = visual.ShapeStim(
        win=win, name='ternusExampleRight_2',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    FixationCrossExample_2 = visual.ShapeStim(
        win=win, name='FixationCrossExample_2', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "blank800" ---
    FixationCrossBlank800 = visual.ShapeStim(
        win=win, name='FixationCrossBlank800', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "elementMotionExplained" ---
    keyElementMotionExplained = keyboard.Keyboard(deviceName='keyElementMotionExplained')
    textElementMotion = visual.TextStim(win=win, name='textElementMotion',
        text='Bei sehr kurzen Unterbrechungen entsteht häufig der Eindruck einer Elementbewegung.\n\n\n\nDabei scheinen die beiden mittleren Kreise an ihrer Position zu bleiben, während links ein Kreis verschwindet und rechts ein neuer Kreis erscheint.\n\n\n\nDrücke die Leertaste, um fortzufahren.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "exampleFrame1" ---
    ternusExampleLeft = visual.ShapeStim(
        win=win, name='ternusExampleLeft',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    ternusExampleMiddle = visual.ShapeStim(
        win=win, name='ternusExampleMiddle',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    ternusExampleRight = visual.ShapeStim(
        win=win, name='ternusExampleRight',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    FixationCrossExample = visual.ShapeStim(
        win=win, name='FixationCrossExample', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "blank160" ---
    FixationCrossBlank160 = visual.ShapeStim(
        win=win, name='FixationCrossBlank160', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "exampleFrame2" ---
    ternusExampleLeft_2 = visual.ShapeStim(
        win=win, name='ternusExampleLeft_2',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    ternusExampleMiddle_2 = visual.ShapeStim(
        win=win, name='ternusExampleMiddle_2',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    ternusExampleRight_2 = visual.ShapeStim(
        win=win, name='ternusExampleRight_2',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    FixationCrossExample_2 = visual.ShapeStim(
        win=win, name='FixationCrossExample_2', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "blank800" ---
    FixationCrossBlank800 = visual.ShapeStim(
        win=win, name='FixationCrossBlank800', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "groupMotionExplained" ---
    keyGroupMotionExplained = keyboard.Keyboard(deviceName='keyGroupMotionExplained')
    textGroupMotion = visual.TextStim(win=win, name='textGroupMotion',
        text="Bei längeren Unterbrechungen entsteht häufig der Eindruck einer Gruppenbewegung.\n\n\n\nDabei scheinen sich alle drei Kreise gemeinsam nach rechts zu bewegen.\n\n\n\nDrücke die Leertaste, um mit den Übungsdurchgängen zu beginnen.\n\nDabei musst du nach jeder Bewegung deinen Eindruck angeben, ob es sich um eine Elementbewegung ('F'-Taste) oder eine Gruppenbewegung ('J'-Taste) handelte.",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "practiceFrame1" ---
    # Run 'Begin Experiment' code from codeObjectParametersPractice
    import random
    
    blockNumber = 0
    ternusDuration = 0.2
    
    # default Frame 2 Ternus colors
    leftFrame2_Color = "black"
    middleFrame2_Color = "black"
    rightFrame2_Color = "black"
    
    colorList = [
        [0,   1, 0.8],   # red
        [45,  1, 0.8],
        [90,  1, 0.8],
        [135, 1, 0.8],
        [180, 1, 0.8],
        [225, 1, 0.8],
        [270, 1, 0.8],
        [315, 1, 0.8]
    ]
    causalColor = random.choice(colorList)
    ternusObjLeftPrac = visual.ShapeStim(
        win=win, name='ternusObjLeftPrac',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    ternusObjMiddlePrac = visual.ShapeStim(
        win=win, name='ternusObjMiddlePrac',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    ternusObjRightPrac = visual.ShapeStim(
        win=win, name='ternusObjRightPrac',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    FixationCrossPrac = visual.ShapeStim(
        win=win, name='FixationCrossPrac', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    portalLeftPrac = visual.Rect(
        win=win, name='portalLeftPrac',units='deg', 
        width=(0.1, 2)[0], height=(0.1, 2)[1],
        ori=0.0, pos=(-4, 2), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='hsv', lineColor='white', fillColor='white',
        opacity=1.0, depth=-5.0, interpolate=True)
    portalRightPrac = visual.Rect(
        win=win, name='portalRightPrac',units='deg', 
        width=(0.1, 2)[0], height=(0.1, 2)[1],
        ori=0.0, pos=(4, 2), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='hsv', lineColor='white', fillColor='white',
        opacity=1.0, depth=-6.0, interpolate=True)
    BarrierPrac = visual.Rect(
        win=win, name='BarrierPrac',units='deg', 
        width=(0.1, 2)[0], height=(0.1, 2)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='hsv', lineColor='white', fillColor='white',
        opacity=1.0, depth=-7.0, interpolate=True)
    
    # --- Initialize components for Routine "blankISI_Practice" ---
    FixationCrossBlankPractice = visual.ShapeStim(
        win=win, name='FixationCrossBlankPractice', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "practiceFrame2" ---
    keyPractice = keyboard.Keyboard(deviceName='keyPractice')
    ternusObjLeftPrac_2 = visual.ShapeStim(
        win=win, name='ternusObjLeftPrac_2',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='hsv', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    ternusObjMiddlePrac_2 = visual.ShapeStim(
        win=win, name='ternusObjMiddlePrac_2',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    ternusObjRightPrac_2 = visual.ShapeStim(
        win=win, name='ternusObjRightPrac_2',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='hsv', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    FixationCrossPrac_2 = visual.ShapeStim(
        win=win, name='FixationCrossPrac_2', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    portalLeftPrac_2 = visual.Rect(
        win=win, name='portalLeftPrac_2',units='deg', 
        width=(0.1, 2)[0], height=(0.1, 2)[1],
        ori=0.0, pos=(-4, 2), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='hsv', lineColor='white', fillColor='white',
        opacity=1.0, depth=-5.0, interpolate=True)
    portalRightPrac_2 = visual.Rect(
        win=win, name='portalRightPrac_2',units='deg', 
        width=(0.1, 2)[0], height=(0.1, 2)[1],
        ori=0.0, pos=(4, 2), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='hsv', lineColor='white', fillColor='white',
        opacity=1.0, depth=-6.0, interpolate=True)
    BarrierPrac_2 = visual.Rect(
        win=win, name='BarrierPrac_2',units='deg', 
        width=(0.1, 2)[0], height=(0.1, 2)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='hsv', lineColor='white', fillColor='white',
        opacity=1.0, depth=-7.0, interpolate=True)
    
    # --- Initialize components for Routine "blank800" ---
    FixationCrossBlank800 = visual.ShapeStim(
        win=win, name='FixationCrossBlank800', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "StartScreen" ---
    keyStart = keyboard.Keyboard(deviceName='keyStart')
    textStartMessage = visual.TextStim(win=win, name='textStartMessage',
        text="Hiermit startet das Experiment!\n\nDu musst 10 Blocks absolvieren.\nGib nach jeder Bewegung an, ob es sich um eine Elementbewegung ('F'-Taste) oder eine Gruppenbewegung ('J'-Taste) handelte.\n\nDrücke die Leertaste, um zu beginnen.",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "ternusFrame1" ---
    # Run 'Begin Experiment' code from codeObjectParameters
    barrierPos = (2,2)
    
    # default Frame 2 Ternus colors
    leftFrame2_Color = "black"
    middleFrame2_Color = "black"
    rightFrame2_Color = "black"
    
    ternusObjLeft = visual.ShapeStim(
        win=win, name='ternusObjLeft',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    ternusObjMiddle = visual.ShapeStim(
        win=win, name='ternusObjMiddle',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    ternusObjRight = visual.ShapeStim(
        win=win, name='ternusObjRight',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    FixationCross = visual.ShapeStim(
        win=win, name='FixationCross', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    portalLeft = visual.Rect(
        win=win, name='portalLeft',units='deg', 
        width=(0.1, 2)[0], height=(0.1, 2)[1],
        ori=0.0, pos=(-4, 2), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-5.0, interpolate=True)
    portalRight = visual.Rect(
        win=win, name='portalRight',units='deg', 
        width=(0.1, 2)[0], height=(0.1, 2)[1],
        ori=0.0, pos=(4, 2), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-6.0, interpolate=True)
    Barrier = visual.Rect(
        win=win, name='Barrier',units='deg', 
        width=(0.1, 2)[0], height=(0.1, 2)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-7.0, interpolate=True)
    
    # --- Initialize components for Routine "blankISI" ---
    FixationCrossBlank = visual.ShapeStim(
        win=win, name='FixationCrossBlank', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "ternusFrame2" ---
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    ternusObjLeft_2 = visual.ShapeStim(
        win=win, name='ternusObjLeft_2',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    ternusObjMiddle_2 = visual.ShapeStim(
        win=win, name='ternusObjMiddle_2',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=middleFrame2_Color, fillColor=middleFrame2_Color,
        opacity=None, depth=-2.0, interpolate=True)
    ternusObjRight_2 = visual.ShapeStim(
        win=win, name='ternusObjRight_2',units='deg', 
        size=(1.6), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    FixationCross_2 = visual.ShapeStim(
        win=win, name='FixationCross_2', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    portalLeft_2 = visual.Rect(
        win=win, name='portalLeft_2',units='deg', 
        width=(0.1, 2)[0], height=(0.1, 2)[1],
        ori=0.0, pos=(-4, 2), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-5.0, interpolate=True)
    portalRight_2 = visual.Rect(
        win=win, name='portalRight_2',units='deg', 
        width=(0.1, 2)[0], height=(0.1, 2)[1],
        ori=0.0, pos=(4, 2), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-6.0, interpolate=True)
    Barrier_2 = visual.Rect(
        win=win, name='Barrier_2',units='deg', 
        width=(0.1, 2)[0], height=(0.1, 2)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-7.0, interpolate=True)
    
    # --- Initialize components for Routine "blank800" ---
    FixationCrossBlank800 = visual.ShapeStim(
        win=win, name='FixationCrossBlank800', vertices='cross',units='deg', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "blockBreak" ---
    keyBlockContinue = keyboard.Keyboard(deviceName='keyBlockContinue')
    textBlockInstructions = visual.TextStim(win=win, name='textBlockInstructions',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "EndScreen" ---
    textEnd = visual.TextStim(win=win, name='textEnd',
        text='Das Experiment ist abgeschlossen! \n\nDu kannst den Arbeitsplatz jetzt verlassen und dem Versuchsleiter Bescheid geben.\n\nVielen Dank für die Teilnahme!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "InstructionScreen" ---
    # create an object to store info about Routine InstructionScreen
    InstructionScreen = data.Routine(
        name='InstructionScreen',
        components=[keyInstructionContinue, textInstructions],
    )
    InstructionScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for keyInstructionContinue
    keyInstructionContinue.keys = []
    keyInstructionContinue.rt = []
    _keyInstructionContinue_allKeys = []
    # store start times for InstructionScreen
    InstructionScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    InstructionScreen.tStart = globalClock.getTime(format='float')
    InstructionScreen.status = STARTED
    thisExp.addData('InstructionScreen.started', InstructionScreen.tStart)
    InstructionScreen.maxDuration = None
    # keep track of which components have finished
    InstructionScreenComponents = InstructionScreen.components
    for thisComponent in InstructionScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "InstructionScreen" ---
    InstructionScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *keyInstructionContinue* updates
        waitOnFlip = False
        
        # if keyInstructionContinue is starting this frame...
        if keyInstructionContinue.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
            # keep track of start time/frame for later
            keyInstructionContinue.frameNStart = frameN  # exact frame index
            keyInstructionContinue.tStart = t  # local t and not account for scr refresh
            keyInstructionContinue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyInstructionContinue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyInstructionContinue.started')
            # update status
            keyInstructionContinue.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyInstructionContinue.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyInstructionContinue.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyInstructionContinue.status == STARTED and not waitOnFlip:
            theseKeys = keyInstructionContinue.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _keyInstructionContinue_allKeys.extend(theseKeys)
            if len(_keyInstructionContinue_allKeys):
                keyInstructionContinue.keys = _keyInstructionContinue_allKeys[-1].name  # just the last key pressed
                keyInstructionContinue.rt = _keyInstructionContinue_allKeys[-1].rt
                keyInstructionContinue.duration = _keyInstructionContinue_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *textInstructions* updates
        
        # if textInstructions is starting this frame...
        if textInstructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textInstructions.frameNStart = frameN  # exact frame index
            textInstructions.tStart = t  # local t and not account for scr refresh
            textInstructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textInstructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textInstructions.started')
            # update status
            textInstructions.status = STARTED
            textInstructions.setAutoDraw(True)
        
        # if textInstructions is active this frame...
        if textInstructions.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=InstructionScreen,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            InstructionScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in InstructionScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "InstructionScreen" ---
    for thisComponent in InstructionScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for InstructionScreen
    InstructionScreen.tStop = globalClock.getTime(format='float')
    InstructionScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('InstructionScreen.stopped', InstructionScreen.tStop)
    # check responses
    if keyInstructionContinue.keys in ['', [], None]:  # No response was made
        keyInstructionContinue.keys = None
    thisExp.addData('keyInstructionContinue.keys',keyInstructionContinue.keys)
    if keyInstructionContinue.keys != None:  # we had a response
        thisExp.addData('keyInstructionContinue.rt', keyInstructionContinue.rt)
        thisExp.addData('keyInstructionContinue.duration', keyInstructionContinue.duration)
    thisExp.nextEntry()
    # the Routine "InstructionScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    elementMotionDemo = data.TrialHandler2(
        name='elementMotionDemo',
        nReps=4.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(elementMotionDemo)  # add the loop to the experiment
    thisElementMotionDemo = elementMotionDemo.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisElementMotionDemo.rgb)
    if thisElementMotionDemo != None:
        for paramName in thisElementMotionDemo:
            globals()[paramName] = thisElementMotionDemo[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisElementMotionDemo in elementMotionDemo:
        elementMotionDemo.status = STARTED
        if hasattr(thisElementMotionDemo, 'status'):
            thisElementMotionDemo.status = STARTED
        currentLoop = elementMotionDemo
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisElementMotionDemo.rgb)
        if thisElementMotionDemo != None:
            for paramName in thisElementMotionDemo:
                globals()[paramName] = thisElementMotionDemo[paramName]
        
        # --- Prepare to start Routine "exampleFrame1" ---
        # create an object to store info about Routine exampleFrame1
        exampleFrame1 = data.Routine(
            name='exampleFrame1',
            components=[ternusExampleLeft, ternusExampleMiddle, ternusExampleRight, FixationCrossExample],
        )
        exampleFrame1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeMovementDirection
        direction = np.random.choice(["LR", "RL"])
        thisExp.addData("direction", direction)
        
        if direction == "LR":
            leftFrame1_Pos = (-3, 2)
            middleFrame1_Pos = (-1, 2)
            rightFrame1_Pos = (1, 2)
        
            leftFrame2_Pos = (-1, 2)
            middleFrame2_Pos = (1, 2)
            rightFrame2_Pos = (3, 2)
        
        elif direction == "RL":
            leftFrame1_Pos = (-1, 2)
            middleFrame1_Pos = (1, 2)
            rightFrame1_Pos = (3, 2)
        
            leftFrame2_Pos = (-3, 2)
            middleFrame2_Pos = (-1, 2)
            rightFrame2_Pos = (1, 2)
        ternusExampleLeft.setPos(leftFrame1_Pos)
        ternusExampleMiddle.setPos(middleFrame1_Pos)
        ternusExampleRight.setPos(rightFrame1_Pos)
        # store start times for exampleFrame1
        exampleFrame1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        exampleFrame1.tStart = globalClock.getTime(format='float')
        exampleFrame1.status = STARTED
        thisExp.addData('exampleFrame1.started', exampleFrame1.tStart)
        exampleFrame1.maxDuration = None
        # keep track of which components have finished
        exampleFrame1Components = exampleFrame1.components
        for thisComponent in exampleFrame1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "exampleFrame1" ---
        exampleFrame1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisElementMotionDemo, 'status') and thisElementMotionDemo.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ternusExampleLeft* updates
            
            # if ternusExampleLeft is starting this frame...
            if ternusExampleLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusExampleLeft.frameNStart = frameN  # exact frame index
                ternusExampleLeft.tStart = t  # local t and not account for scr refresh
                ternusExampleLeft.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusExampleLeft, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusExampleLeft.started')
                # update status
                ternusExampleLeft.status = STARTED
                ternusExampleLeft.setAutoDraw(True)
            
            # if ternusExampleLeft is active this frame...
            if ternusExampleLeft.status == STARTED:
                # update params
                pass
            
            # if ternusExampleLeft is stopping this frame...
            if ternusExampleLeft.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusExampleLeft.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusExampleLeft.tStop = t  # not accounting for scr refresh
                    ternusExampleLeft.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusExampleLeft.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusExampleLeft.stopped')
                    # update status
                    ternusExampleLeft.status = FINISHED
                    ternusExampleLeft.setAutoDraw(False)
            
            # *ternusExampleMiddle* updates
            
            # if ternusExampleMiddle is starting this frame...
            if ternusExampleMiddle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusExampleMiddle.frameNStart = frameN  # exact frame index
                ternusExampleMiddle.tStart = t  # local t and not account for scr refresh
                ternusExampleMiddle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusExampleMiddle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusExampleMiddle.started')
                # update status
                ternusExampleMiddle.status = STARTED
                ternusExampleMiddle.setAutoDraw(True)
            
            # if ternusExampleMiddle is active this frame...
            if ternusExampleMiddle.status == STARTED:
                # update params
                pass
            
            # if ternusExampleMiddle is stopping this frame...
            if ternusExampleMiddle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusExampleMiddle.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusExampleMiddle.tStop = t  # not accounting for scr refresh
                    ternusExampleMiddle.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusExampleMiddle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusExampleMiddle.stopped')
                    # update status
                    ternusExampleMiddle.status = FINISHED
                    ternusExampleMiddle.setAutoDraw(False)
            
            # *ternusExampleRight* updates
            
            # if ternusExampleRight is starting this frame...
            if ternusExampleRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusExampleRight.frameNStart = frameN  # exact frame index
                ternusExampleRight.tStart = t  # local t and not account for scr refresh
                ternusExampleRight.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusExampleRight, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusExampleRight.started')
                # update status
                ternusExampleRight.status = STARTED
                ternusExampleRight.setAutoDraw(True)
            
            # if ternusExampleRight is active this frame...
            if ternusExampleRight.status == STARTED:
                # update params
                pass
            
            # if ternusExampleRight is stopping this frame...
            if ternusExampleRight.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusExampleRight.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusExampleRight.tStop = t  # not accounting for scr refresh
                    ternusExampleRight.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusExampleRight.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusExampleRight.stopped')
                    # update status
                    ternusExampleRight.status = FINISHED
                    ternusExampleRight.setAutoDraw(False)
            
            # *FixationCrossExample* updates
            
            # if FixationCrossExample is starting this frame...
            if FixationCrossExample.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCrossExample.frameNStart = frameN  # exact frame index
                FixationCrossExample.tStart = t  # local t and not account for scr refresh
                FixationCrossExample.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCrossExample, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationCrossExample.started')
                # update status
                FixationCrossExample.status = STARTED
                FixationCrossExample.setAutoDraw(True)
            
            # if FixationCrossExample is active this frame...
            if FixationCrossExample.status == STARTED:
                # update params
                pass
            
            # if FixationCrossExample is stopping this frame...
            if FixationCrossExample.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCrossExample.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCrossExample.tStop = t  # not accounting for scr refresh
                    FixationCrossExample.tStopRefresh = tThisFlipGlobal  # on global time
                    FixationCrossExample.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCrossExample.stopped')
                    # update status
                    FixationCrossExample.status = FINISHED
                    FixationCrossExample.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=exampleFrame1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                exampleFrame1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in exampleFrame1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "exampleFrame1" ---
        for thisComponent in exampleFrame1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for exampleFrame1
        exampleFrame1.tStop = globalClock.getTime(format='float')
        exampleFrame1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('exampleFrame1.stopped', exampleFrame1.tStop)
        # the Routine "exampleFrame1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "exampleFrame2" ---
        # create an object to store info about Routine exampleFrame2
        exampleFrame2 = data.Routine(
            name='exampleFrame2',
            components=[ternusExampleLeft_2, ternusExampleMiddle_2, ternusExampleRight_2, FixationCrossExample_2],
        )
        exampleFrame2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        ternusExampleLeft_2.setPos(leftFrame2_Pos)
        ternusExampleMiddle_2.setPos(middleFrame2_Pos)
        ternusExampleRight_2.setPos(rightFrame2_Pos)
        # store start times for exampleFrame2
        exampleFrame2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        exampleFrame2.tStart = globalClock.getTime(format='float')
        exampleFrame2.status = STARTED
        thisExp.addData('exampleFrame2.started', exampleFrame2.tStart)
        exampleFrame2.maxDuration = None
        # keep track of which components have finished
        exampleFrame2Components = exampleFrame2.components
        for thisComponent in exampleFrame2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "exampleFrame2" ---
        exampleFrame2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisElementMotionDemo, 'status') and thisElementMotionDemo.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ternusExampleLeft_2* updates
            
            # if ternusExampleLeft_2 is starting this frame...
            if ternusExampleLeft_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusExampleLeft_2.frameNStart = frameN  # exact frame index
                ternusExampleLeft_2.tStart = t  # local t and not account for scr refresh
                ternusExampleLeft_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusExampleLeft_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusExampleLeft_2.started')
                # update status
                ternusExampleLeft_2.status = STARTED
                ternusExampleLeft_2.setAutoDraw(True)
            
            # if ternusExampleLeft_2 is active this frame...
            if ternusExampleLeft_2.status == STARTED:
                # update params
                pass
            
            # if ternusExampleLeft_2 is stopping this frame...
            if ternusExampleLeft_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusExampleLeft_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusExampleLeft_2.tStop = t  # not accounting for scr refresh
                    ternusExampleLeft_2.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusExampleLeft_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusExampleLeft_2.stopped')
                    # update status
                    ternusExampleLeft_2.status = FINISHED
                    ternusExampleLeft_2.setAutoDraw(False)
            
            # *ternusExampleMiddle_2* updates
            
            # if ternusExampleMiddle_2 is starting this frame...
            if ternusExampleMiddle_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusExampleMiddle_2.frameNStart = frameN  # exact frame index
                ternusExampleMiddle_2.tStart = t  # local t and not account for scr refresh
                ternusExampleMiddle_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusExampleMiddle_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusExampleMiddle_2.started')
                # update status
                ternusExampleMiddle_2.status = STARTED
                ternusExampleMiddle_2.setAutoDraw(True)
            
            # if ternusExampleMiddle_2 is active this frame...
            if ternusExampleMiddle_2.status == STARTED:
                # update params
                pass
            
            # if ternusExampleMiddle_2 is stopping this frame...
            if ternusExampleMiddle_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusExampleMiddle_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusExampleMiddle_2.tStop = t  # not accounting for scr refresh
                    ternusExampleMiddle_2.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusExampleMiddle_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusExampleMiddle_2.stopped')
                    # update status
                    ternusExampleMiddle_2.status = FINISHED
                    ternusExampleMiddle_2.setAutoDraw(False)
            
            # *ternusExampleRight_2* updates
            
            # if ternusExampleRight_2 is starting this frame...
            if ternusExampleRight_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusExampleRight_2.frameNStart = frameN  # exact frame index
                ternusExampleRight_2.tStart = t  # local t and not account for scr refresh
                ternusExampleRight_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusExampleRight_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusExampleRight_2.started')
                # update status
                ternusExampleRight_2.status = STARTED
                ternusExampleRight_2.setAutoDraw(True)
            
            # if ternusExampleRight_2 is active this frame...
            if ternusExampleRight_2.status == STARTED:
                # update params
                pass
            
            # if ternusExampleRight_2 is stopping this frame...
            if ternusExampleRight_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusExampleRight_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusExampleRight_2.tStop = t  # not accounting for scr refresh
                    ternusExampleRight_2.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusExampleRight_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusExampleRight_2.stopped')
                    # update status
                    ternusExampleRight_2.status = FINISHED
                    ternusExampleRight_2.setAutoDraw(False)
            
            # *FixationCrossExample_2* updates
            
            # if FixationCrossExample_2 is starting this frame...
            if FixationCrossExample_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCrossExample_2.frameNStart = frameN  # exact frame index
                FixationCrossExample_2.tStart = t  # local t and not account for scr refresh
                FixationCrossExample_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCrossExample_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationCrossExample_2.started')
                # update status
                FixationCrossExample_2.status = STARTED
                FixationCrossExample_2.setAutoDraw(True)
            
            # if FixationCrossExample_2 is active this frame...
            if FixationCrossExample_2.status == STARTED:
                # update params
                pass
            
            # if FixationCrossExample_2 is stopping this frame...
            if FixationCrossExample_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCrossExample_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCrossExample_2.tStop = t  # not accounting for scr refresh
                    FixationCrossExample_2.tStopRefresh = tThisFlipGlobal  # on global time
                    FixationCrossExample_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCrossExample_2.stopped')
                    # update status
                    FixationCrossExample_2.status = FINISHED
                    FixationCrossExample_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=exampleFrame2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                exampleFrame2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in exampleFrame2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "exampleFrame2" ---
        for thisComponent in exampleFrame2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for exampleFrame2
        exampleFrame2.tStop = globalClock.getTime(format='float')
        exampleFrame2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('exampleFrame2.stopped', exampleFrame2.tStop)
        # the Routine "exampleFrame2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank800" ---
        # create an object to store info about Routine blank800
        blank800 = data.Routine(
            name='blank800',
            components=[FixationCrossBlank800],
        )
        blank800.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank800
        blank800.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank800.tStart = globalClock.getTime(format='float')
        blank800.status = STARTED
        thisExp.addData('blank800.started', blank800.tStart)
        blank800.maxDuration = None
        # keep track of which components have finished
        blank800Components = blank800.components
        for thisComponent in blank800.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank800" ---
        blank800.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.8:
            # if trial has changed, end Routine now
            if hasattr(thisElementMotionDemo, 'status') and thisElementMotionDemo.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *FixationCrossBlank800* updates
            
            # if FixationCrossBlank800 is starting this frame...
            if FixationCrossBlank800.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCrossBlank800.frameNStart = frameN  # exact frame index
                FixationCrossBlank800.tStart = t  # local t and not account for scr refresh
                FixationCrossBlank800.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCrossBlank800, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationCrossBlank800.started')
                # update status
                FixationCrossBlank800.status = STARTED
                FixationCrossBlank800.setAutoDraw(True)
            
            # if FixationCrossBlank800 is active this frame...
            if FixationCrossBlank800.status == STARTED:
                # update params
                pass
            
            # if FixationCrossBlank800 is stopping this frame...
            if FixationCrossBlank800.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCrossBlank800.tStartRefresh + 0.8-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCrossBlank800.tStop = t  # not accounting for scr refresh
                    FixationCrossBlank800.tStopRefresh = tThisFlipGlobal  # on global time
                    FixationCrossBlank800.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCrossBlank800.stopped')
                    # update status
                    FixationCrossBlank800.status = FINISHED
                    FixationCrossBlank800.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank800,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank800.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank800.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank800" ---
        for thisComponent in blank800.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank800
        blank800.tStop = globalClock.getTime(format='float')
        blank800.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank800.stopped', blank800.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank800.maxDurationReached:
            routineTimer.addTime(-blank800.maxDuration)
        elif blank800.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.800000)
        # mark thisElementMotionDemo as finished
        if hasattr(thisElementMotionDemo, 'status'):
            thisElementMotionDemo.status = FINISHED
        # if awaiting a pause, pause now
        if elementMotionDemo.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            elementMotionDemo.status = STARTED
        thisExp.nextEntry()
        
    # completed 4.0 repeats of 'elementMotionDemo'
    elementMotionDemo.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "elementMotionExplained" ---
    # create an object to store info about Routine elementMotionExplained
    elementMotionExplained = data.Routine(
        name='elementMotionExplained',
        components=[keyElementMotionExplained, textElementMotion],
    )
    elementMotionExplained.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for keyElementMotionExplained
    keyElementMotionExplained.keys = []
    keyElementMotionExplained.rt = []
    _keyElementMotionExplained_allKeys = []
    # store start times for elementMotionExplained
    elementMotionExplained.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    elementMotionExplained.tStart = globalClock.getTime(format='float')
    elementMotionExplained.status = STARTED
    thisExp.addData('elementMotionExplained.started', elementMotionExplained.tStart)
    elementMotionExplained.maxDuration = None
    # keep track of which components have finished
    elementMotionExplainedComponents = elementMotionExplained.components
    for thisComponent in elementMotionExplained.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "elementMotionExplained" ---
    elementMotionExplained.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *keyElementMotionExplained* updates
        waitOnFlip = False
        
        # if keyElementMotionExplained is starting this frame...
        if keyElementMotionExplained.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            keyElementMotionExplained.frameNStart = frameN  # exact frame index
            keyElementMotionExplained.tStart = t  # local t and not account for scr refresh
            keyElementMotionExplained.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyElementMotionExplained, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyElementMotionExplained.started')
            # update status
            keyElementMotionExplained.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyElementMotionExplained.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyElementMotionExplained.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyElementMotionExplained.status == STARTED and not waitOnFlip:
            theseKeys = keyElementMotionExplained.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _keyElementMotionExplained_allKeys.extend(theseKeys)
            if len(_keyElementMotionExplained_allKeys):
                keyElementMotionExplained.keys = _keyElementMotionExplained_allKeys[-1].name  # just the last key pressed
                keyElementMotionExplained.rt = _keyElementMotionExplained_allKeys[-1].rt
                keyElementMotionExplained.duration = _keyElementMotionExplained_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *textElementMotion* updates
        
        # if textElementMotion is starting this frame...
        if textElementMotion.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textElementMotion.frameNStart = frameN  # exact frame index
            textElementMotion.tStart = t  # local t and not account for scr refresh
            textElementMotion.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textElementMotion, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textElementMotion.started')
            # update status
            textElementMotion.status = STARTED
            textElementMotion.setAutoDraw(True)
        
        # if textElementMotion is active this frame...
        if textElementMotion.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=elementMotionExplained,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            elementMotionExplained.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in elementMotionExplained.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "elementMotionExplained" ---
    for thisComponent in elementMotionExplained.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for elementMotionExplained
    elementMotionExplained.tStop = globalClock.getTime(format='float')
    elementMotionExplained.tStopRefresh = tThisFlipGlobal
    thisExp.addData('elementMotionExplained.stopped', elementMotionExplained.tStop)
    # check responses
    if keyElementMotionExplained.keys in ['', [], None]:  # No response was made
        keyElementMotionExplained.keys = None
    thisExp.addData('keyElementMotionExplained.keys',keyElementMotionExplained.keys)
    if keyElementMotionExplained.keys != None:  # we had a response
        thisExp.addData('keyElementMotionExplained.rt', keyElementMotionExplained.rt)
        thisExp.addData('keyElementMotionExplained.duration', keyElementMotionExplained.duration)
    thisExp.nextEntry()
    # the Routine "elementMotionExplained" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    groupMotionDemo = data.TrialHandler2(
        name='groupMotionDemo',
        nReps=4.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(groupMotionDemo)  # add the loop to the experiment
    thisGroupMotionDemo = groupMotionDemo.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisGroupMotionDemo.rgb)
    if thisGroupMotionDemo != None:
        for paramName in thisGroupMotionDemo:
            globals()[paramName] = thisGroupMotionDemo[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisGroupMotionDemo in groupMotionDemo:
        groupMotionDemo.status = STARTED
        if hasattr(thisGroupMotionDemo, 'status'):
            thisGroupMotionDemo.status = STARTED
        currentLoop = groupMotionDemo
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisGroupMotionDemo.rgb)
        if thisGroupMotionDemo != None:
            for paramName in thisGroupMotionDemo:
                globals()[paramName] = thisGroupMotionDemo[paramName]
        
        # --- Prepare to start Routine "exampleFrame1" ---
        # create an object to store info about Routine exampleFrame1
        exampleFrame1 = data.Routine(
            name='exampleFrame1',
            components=[ternusExampleLeft, ternusExampleMiddle, ternusExampleRight, FixationCrossExample],
        )
        exampleFrame1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeMovementDirection
        direction = np.random.choice(["LR", "RL"])
        thisExp.addData("direction", direction)
        
        if direction == "LR":
            leftFrame1_Pos = (-3, 2)
            middleFrame1_Pos = (-1, 2)
            rightFrame1_Pos = (1, 2)
        
            leftFrame2_Pos = (-1, 2)
            middleFrame2_Pos = (1, 2)
            rightFrame2_Pos = (3, 2)
        
        elif direction == "RL":
            leftFrame1_Pos = (-1, 2)
            middleFrame1_Pos = (1, 2)
            rightFrame1_Pos = (3, 2)
        
            leftFrame2_Pos = (-3, 2)
            middleFrame2_Pos = (-1, 2)
            rightFrame2_Pos = (1, 2)
        ternusExampleLeft.setPos(leftFrame1_Pos)
        ternusExampleMiddle.setPos(middleFrame1_Pos)
        ternusExampleRight.setPos(rightFrame1_Pos)
        # store start times for exampleFrame1
        exampleFrame1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        exampleFrame1.tStart = globalClock.getTime(format='float')
        exampleFrame1.status = STARTED
        thisExp.addData('exampleFrame1.started', exampleFrame1.tStart)
        exampleFrame1.maxDuration = None
        # keep track of which components have finished
        exampleFrame1Components = exampleFrame1.components
        for thisComponent in exampleFrame1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "exampleFrame1" ---
        exampleFrame1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisGroupMotionDemo, 'status') and thisGroupMotionDemo.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ternusExampleLeft* updates
            
            # if ternusExampleLeft is starting this frame...
            if ternusExampleLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusExampleLeft.frameNStart = frameN  # exact frame index
                ternusExampleLeft.tStart = t  # local t and not account for scr refresh
                ternusExampleLeft.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusExampleLeft, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusExampleLeft.started')
                # update status
                ternusExampleLeft.status = STARTED
                ternusExampleLeft.setAutoDraw(True)
            
            # if ternusExampleLeft is active this frame...
            if ternusExampleLeft.status == STARTED:
                # update params
                pass
            
            # if ternusExampleLeft is stopping this frame...
            if ternusExampleLeft.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusExampleLeft.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusExampleLeft.tStop = t  # not accounting for scr refresh
                    ternusExampleLeft.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusExampleLeft.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusExampleLeft.stopped')
                    # update status
                    ternusExampleLeft.status = FINISHED
                    ternusExampleLeft.setAutoDraw(False)
            
            # *ternusExampleMiddle* updates
            
            # if ternusExampleMiddle is starting this frame...
            if ternusExampleMiddle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusExampleMiddle.frameNStart = frameN  # exact frame index
                ternusExampleMiddle.tStart = t  # local t and not account for scr refresh
                ternusExampleMiddle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusExampleMiddle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusExampleMiddle.started')
                # update status
                ternusExampleMiddle.status = STARTED
                ternusExampleMiddle.setAutoDraw(True)
            
            # if ternusExampleMiddle is active this frame...
            if ternusExampleMiddle.status == STARTED:
                # update params
                pass
            
            # if ternusExampleMiddle is stopping this frame...
            if ternusExampleMiddle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusExampleMiddle.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusExampleMiddle.tStop = t  # not accounting for scr refresh
                    ternusExampleMiddle.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusExampleMiddle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusExampleMiddle.stopped')
                    # update status
                    ternusExampleMiddle.status = FINISHED
                    ternusExampleMiddle.setAutoDraw(False)
            
            # *ternusExampleRight* updates
            
            # if ternusExampleRight is starting this frame...
            if ternusExampleRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusExampleRight.frameNStart = frameN  # exact frame index
                ternusExampleRight.tStart = t  # local t and not account for scr refresh
                ternusExampleRight.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusExampleRight, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusExampleRight.started')
                # update status
                ternusExampleRight.status = STARTED
                ternusExampleRight.setAutoDraw(True)
            
            # if ternusExampleRight is active this frame...
            if ternusExampleRight.status == STARTED:
                # update params
                pass
            
            # if ternusExampleRight is stopping this frame...
            if ternusExampleRight.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusExampleRight.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusExampleRight.tStop = t  # not accounting for scr refresh
                    ternusExampleRight.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusExampleRight.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusExampleRight.stopped')
                    # update status
                    ternusExampleRight.status = FINISHED
                    ternusExampleRight.setAutoDraw(False)
            
            # *FixationCrossExample* updates
            
            # if FixationCrossExample is starting this frame...
            if FixationCrossExample.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCrossExample.frameNStart = frameN  # exact frame index
                FixationCrossExample.tStart = t  # local t and not account for scr refresh
                FixationCrossExample.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCrossExample, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationCrossExample.started')
                # update status
                FixationCrossExample.status = STARTED
                FixationCrossExample.setAutoDraw(True)
            
            # if FixationCrossExample is active this frame...
            if FixationCrossExample.status == STARTED:
                # update params
                pass
            
            # if FixationCrossExample is stopping this frame...
            if FixationCrossExample.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCrossExample.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCrossExample.tStop = t  # not accounting for scr refresh
                    FixationCrossExample.tStopRefresh = tThisFlipGlobal  # on global time
                    FixationCrossExample.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCrossExample.stopped')
                    # update status
                    FixationCrossExample.status = FINISHED
                    FixationCrossExample.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=exampleFrame1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                exampleFrame1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in exampleFrame1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "exampleFrame1" ---
        for thisComponent in exampleFrame1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for exampleFrame1
        exampleFrame1.tStop = globalClock.getTime(format='float')
        exampleFrame1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('exampleFrame1.stopped', exampleFrame1.tStop)
        # the Routine "exampleFrame1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank160" ---
        # create an object to store info about Routine blank160
        blank160 = data.Routine(
            name='blank160',
            components=[FixationCrossBlank160],
        )
        blank160.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank160
        blank160.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank160.tStart = globalClock.getTime(format='float')
        blank160.status = STARTED
        thisExp.addData('blank160.started', blank160.tStart)
        blank160.maxDuration = None
        # keep track of which components have finished
        blank160Components = blank160.components
        for thisComponent in blank160.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank160" ---
        blank160.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.16:
            # if trial has changed, end Routine now
            if hasattr(thisGroupMotionDemo, 'status') and thisGroupMotionDemo.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *FixationCrossBlank160* updates
            
            # if FixationCrossBlank160 is starting this frame...
            if FixationCrossBlank160.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCrossBlank160.frameNStart = frameN  # exact frame index
                FixationCrossBlank160.tStart = t  # local t and not account for scr refresh
                FixationCrossBlank160.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCrossBlank160, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationCrossBlank160.started')
                # update status
                FixationCrossBlank160.status = STARTED
                FixationCrossBlank160.setAutoDraw(True)
            
            # if FixationCrossBlank160 is active this frame...
            if FixationCrossBlank160.status == STARTED:
                # update params
                pass
            
            # if FixationCrossBlank160 is stopping this frame...
            if FixationCrossBlank160.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCrossBlank160.tStartRefresh + 0.16-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCrossBlank160.tStop = t  # not accounting for scr refresh
                    FixationCrossBlank160.tStopRefresh = tThisFlipGlobal  # on global time
                    FixationCrossBlank160.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCrossBlank160.stopped')
                    # update status
                    FixationCrossBlank160.status = FINISHED
                    FixationCrossBlank160.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank160,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank160.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank160.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank160" ---
        for thisComponent in blank160.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank160
        blank160.tStop = globalClock.getTime(format='float')
        blank160.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank160.stopped', blank160.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank160.maxDurationReached:
            routineTimer.addTime(-blank160.maxDuration)
        elif blank160.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.160000)
        
        # --- Prepare to start Routine "exampleFrame2" ---
        # create an object to store info about Routine exampleFrame2
        exampleFrame2 = data.Routine(
            name='exampleFrame2',
            components=[ternusExampleLeft_2, ternusExampleMiddle_2, ternusExampleRight_2, FixationCrossExample_2],
        )
        exampleFrame2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        ternusExampleLeft_2.setPos(leftFrame2_Pos)
        ternusExampleMiddle_2.setPos(middleFrame2_Pos)
        ternusExampleRight_2.setPos(rightFrame2_Pos)
        # store start times for exampleFrame2
        exampleFrame2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        exampleFrame2.tStart = globalClock.getTime(format='float')
        exampleFrame2.status = STARTED
        thisExp.addData('exampleFrame2.started', exampleFrame2.tStart)
        exampleFrame2.maxDuration = None
        # keep track of which components have finished
        exampleFrame2Components = exampleFrame2.components
        for thisComponent in exampleFrame2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "exampleFrame2" ---
        exampleFrame2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisGroupMotionDemo, 'status') and thisGroupMotionDemo.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ternusExampleLeft_2* updates
            
            # if ternusExampleLeft_2 is starting this frame...
            if ternusExampleLeft_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusExampleLeft_2.frameNStart = frameN  # exact frame index
                ternusExampleLeft_2.tStart = t  # local t and not account for scr refresh
                ternusExampleLeft_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusExampleLeft_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusExampleLeft_2.started')
                # update status
                ternusExampleLeft_2.status = STARTED
                ternusExampleLeft_2.setAutoDraw(True)
            
            # if ternusExampleLeft_2 is active this frame...
            if ternusExampleLeft_2.status == STARTED:
                # update params
                pass
            
            # if ternusExampleLeft_2 is stopping this frame...
            if ternusExampleLeft_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusExampleLeft_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusExampleLeft_2.tStop = t  # not accounting for scr refresh
                    ternusExampleLeft_2.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusExampleLeft_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusExampleLeft_2.stopped')
                    # update status
                    ternusExampleLeft_2.status = FINISHED
                    ternusExampleLeft_2.setAutoDraw(False)
            
            # *ternusExampleMiddle_2* updates
            
            # if ternusExampleMiddle_2 is starting this frame...
            if ternusExampleMiddle_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusExampleMiddle_2.frameNStart = frameN  # exact frame index
                ternusExampleMiddle_2.tStart = t  # local t and not account for scr refresh
                ternusExampleMiddle_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusExampleMiddle_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusExampleMiddle_2.started')
                # update status
                ternusExampleMiddle_2.status = STARTED
                ternusExampleMiddle_2.setAutoDraw(True)
            
            # if ternusExampleMiddle_2 is active this frame...
            if ternusExampleMiddle_2.status == STARTED:
                # update params
                pass
            
            # if ternusExampleMiddle_2 is stopping this frame...
            if ternusExampleMiddle_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusExampleMiddle_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusExampleMiddle_2.tStop = t  # not accounting for scr refresh
                    ternusExampleMiddle_2.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusExampleMiddle_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusExampleMiddle_2.stopped')
                    # update status
                    ternusExampleMiddle_2.status = FINISHED
                    ternusExampleMiddle_2.setAutoDraw(False)
            
            # *ternusExampleRight_2* updates
            
            # if ternusExampleRight_2 is starting this frame...
            if ternusExampleRight_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusExampleRight_2.frameNStart = frameN  # exact frame index
                ternusExampleRight_2.tStart = t  # local t and not account for scr refresh
                ternusExampleRight_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusExampleRight_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusExampleRight_2.started')
                # update status
                ternusExampleRight_2.status = STARTED
                ternusExampleRight_2.setAutoDraw(True)
            
            # if ternusExampleRight_2 is active this frame...
            if ternusExampleRight_2.status == STARTED:
                # update params
                pass
            
            # if ternusExampleRight_2 is stopping this frame...
            if ternusExampleRight_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusExampleRight_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusExampleRight_2.tStop = t  # not accounting for scr refresh
                    ternusExampleRight_2.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusExampleRight_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusExampleRight_2.stopped')
                    # update status
                    ternusExampleRight_2.status = FINISHED
                    ternusExampleRight_2.setAutoDraw(False)
            
            # *FixationCrossExample_2* updates
            
            # if FixationCrossExample_2 is starting this frame...
            if FixationCrossExample_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCrossExample_2.frameNStart = frameN  # exact frame index
                FixationCrossExample_2.tStart = t  # local t and not account for scr refresh
                FixationCrossExample_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCrossExample_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationCrossExample_2.started')
                # update status
                FixationCrossExample_2.status = STARTED
                FixationCrossExample_2.setAutoDraw(True)
            
            # if FixationCrossExample_2 is active this frame...
            if FixationCrossExample_2.status == STARTED:
                # update params
                pass
            
            # if FixationCrossExample_2 is stopping this frame...
            if FixationCrossExample_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCrossExample_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCrossExample_2.tStop = t  # not accounting for scr refresh
                    FixationCrossExample_2.tStopRefresh = tThisFlipGlobal  # on global time
                    FixationCrossExample_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCrossExample_2.stopped')
                    # update status
                    FixationCrossExample_2.status = FINISHED
                    FixationCrossExample_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=exampleFrame2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                exampleFrame2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in exampleFrame2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "exampleFrame2" ---
        for thisComponent in exampleFrame2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for exampleFrame2
        exampleFrame2.tStop = globalClock.getTime(format='float')
        exampleFrame2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('exampleFrame2.stopped', exampleFrame2.tStop)
        # the Routine "exampleFrame2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank800" ---
        # create an object to store info about Routine blank800
        blank800 = data.Routine(
            name='blank800',
            components=[FixationCrossBlank800],
        )
        blank800.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank800
        blank800.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank800.tStart = globalClock.getTime(format='float')
        blank800.status = STARTED
        thisExp.addData('blank800.started', blank800.tStart)
        blank800.maxDuration = None
        # keep track of which components have finished
        blank800Components = blank800.components
        for thisComponent in blank800.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank800" ---
        blank800.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.8:
            # if trial has changed, end Routine now
            if hasattr(thisGroupMotionDemo, 'status') and thisGroupMotionDemo.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *FixationCrossBlank800* updates
            
            # if FixationCrossBlank800 is starting this frame...
            if FixationCrossBlank800.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCrossBlank800.frameNStart = frameN  # exact frame index
                FixationCrossBlank800.tStart = t  # local t and not account for scr refresh
                FixationCrossBlank800.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCrossBlank800, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationCrossBlank800.started')
                # update status
                FixationCrossBlank800.status = STARTED
                FixationCrossBlank800.setAutoDraw(True)
            
            # if FixationCrossBlank800 is active this frame...
            if FixationCrossBlank800.status == STARTED:
                # update params
                pass
            
            # if FixationCrossBlank800 is stopping this frame...
            if FixationCrossBlank800.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCrossBlank800.tStartRefresh + 0.8-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCrossBlank800.tStop = t  # not accounting for scr refresh
                    FixationCrossBlank800.tStopRefresh = tThisFlipGlobal  # on global time
                    FixationCrossBlank800.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCrossBlank800.stopped')
                    # update status
                    FixationCrossBlank800.status = FINISHED
                    FixationCrossBlank800.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank800,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank800.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank800.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank800" ---
        for thisComponent in blank800.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank800
        blank800.tStop = globalClock.getTime(format='float')
        blank800.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank800.stopped', blank800.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank800.maxDurationReached:
            routineTimer.addTime(-blank800.maxDuration)
        elif blank800.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.800000)
        # mark thisGroupMotionDemo as finished
        if hasattr(thisGroupMotionDemo, 'status'):
            thisGroupMotionDemo.status = FINISHED
        # if awaiting a pause, pause now
        if groupMotionDemo.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            groupMotionDemo.status = STARTED
        thisExp.nextEntry()
        
    # completed 4.0 repeats of 'groupMotionDemo'
    groupMotionDemo.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "groupMotionExplained" ---
    # create an object to store info about Routine groupMotionExplained
    groupMotionExplained = data.Routine(
        name='groupMotionExplained',
        components=[keyGroupMotionExplained, textGroupMotion],
    )
    groupMotionExplained.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for keyGroupMotionExplained
    keyGroupMotionExplained.keys = []
    keyGroupMotionExplained.rt = []
    _keyGroupMotionExplained_allKeys = []
    # store start times for groupMotionExplained
    groupMotionExplained.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    groupMotionExplained.tStart = globalClock.getTime(format='float')
    groupMotionExplained.status = STARTED
    thisExp.addData('groupMotionExplained.started', groupMotionExplained.tStart)
    groupMotionExplained.maxDuration = None
    # keep track of which components have finished
    groupMotionExplainedComponents = groupMotionExplained.components
    for thisComponent in groupMotionExplained.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "groupMotionExplained" ---
    groupMotionExplained.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *keyGroupMotionExplained* updates
        waitOnFlip = False
        
        # if keyGroupMotionExplained is starting this frame...
        if keyGroupMotionExplained.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            keyGroupMotionExplained.frameNStart = frameN  # exact frame index
            keyGroupMotionExplained.tStart = t  # local t and not account for scr refresh
            keyGroupMotionExplained.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyGroupMotionExplained, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyGroupMotionExplained.started')
            # update status
            keyGroupMotionExplained.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyGroupMotionExplained.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyGroupMotionExplained.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyGroupMotionExplained.status == STARTED and not waitOnFlip:
            theseKeys = keyGroupMotionExplained.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _keyGroupMotionExplained_allKeys.extend(theseKeys)
            if len(_keyGroupMotionExplained_allKeys):
                keyGroupMotionExplained.keys = _keyGroupMotionExplained_allKeys[-1].name  # just the last key pressed
                keyGroupMotionExplained.rt = _keyGroupMotionExplained_allKeys[-1].rt
                keyGroupMotionExplained.duration = _keyGroupMotionExplained_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *textGroupMotion* updates
        
        # if textGroupMotion is starting this frame...
        if textGroupMotion.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textGroupMotion.frameNStart = frameN  # exact frame index
            textGroupMotion.tStart = t  # local t and not account for scr refresh
            textGroupMotion.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textGroupMotion, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textGroupMotion.started')
            # update status
            textGroupMotion.status = STARTED
            textGroupMotion.setAutoDraw(True)
        
        # if textGroupMotion is active this frame...
        if textGroupMotion.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=groupMotionExplained,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            groupMotionExplained.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in groupMotionExplained.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "groupMotionExplained" ---
    for thisComponent in groupMotionExplained.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for groupMotionExplained
    groupMotionExplained.tStop = globalClock.getTime(format='float')
    groupMotionExplained.tStopRefresh = tThisFlipGlobal
    thisExp.addData('groupMotionExplained.stopped', groupMotionExplained.tStop)
    # check responses
    if keyGroupMotionExplained.keys in ['', [], None]:  # No response was made
        keyGroupMotionExplained.keys = None
    thisExp.addData('keyGroupMotionExplained.keys',keyGroupMotionExplained.keys)
    if keyGroupMotionExplained.keys != None:  # we had a response
        thisExp.addData('keyGroupMotionExplained.rt', keyGroupMotionExplained.rt)
        thisExp.addData('keyGroupMotionExplained.duration', keyGroupMotionExplained.duration)
    thisExp.nextEntry()
    # the Routine "groupMotionExplained" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practiceLoop = data.TrialHandler2(
        name='practiceLoop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('practiceLoopConditionsTeamprojekt.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(practiceLoop)  # add the loop to the experiment
    thisPracticeLoop = practiceLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPracticeLoop.rgb)
    if thisPracticeLoop != None:
        for paramName in thisPracticeLoop:
            globals()[paramName] = thisPracticeLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPracticeLoop in practiceLoop:
        practiceLoop.status = STARTED
        if hasattr(thisPracticeLoop, 'status'):
            thisPracticeLoop.status = STARTED
        currentLoop = practiceLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPracticeLoop.rgb)
        if thisPracticeLoop != None:
            for paramName in thisPracticeLoop:
                globals()[paramName] = thisPracticeLoop[paramName]
        
        # --- Prepare to start Routine "practiceFrame1" ---
        # create an object to store info about Routine practiceFrame1
        practiceFrame1 = data.Routine(
            name='practiceFrame1',
            components=[ternusObjLeftPrac, ternusObjMiddlePrac, ternusObjRightPrac, FixationCrossPrac, portalLeftPrac, portalRightPrac, BarrierPrac],
        )
        practiceFrame1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeObjectParametersPractice
        colorList = [
            [0,   1, 0.8],   # red
            [45,  1, 0.8],
            [90,  1, 0.8],
            [135, 1, 0.8],
            [180, 1, 0.8],
            [225, 1, 0.8],
            [270, 1, 0.8],
            [315, 1, 0.8]
        ]
        causalColor = random.choice(colorList)
        
        direction = random.choice(["LR", "RL"])
        
        if direction == "LR":
            leftFrame1_Pos = (-3, 2)
            middleFrame1_Pos = (-1, 2)
            rightFrame1_Pos = (1, 2)
        
            leftFrame2_Pos = (-1, 2)
            middleFrame2_Pos = (1, 2)
            rightFrame2_Pos = (3, 2)
        
        elif direction == "RL":
            leftFrame1_Pos = (-1, 2)
            middleFrame1_Pos = (1, 2)
            rightFrame1_Pos = (3, 2)
        
            leftFrame2_Pos = (-3, 2)
            middleFrame2_Pos = (-1, 2)
            rightFrame2_Pos = (1, 2)
        
        # default: invisible causal objects
        portalOpacity = 0
        barrierOpacity = 0
        barrierPos = (2,2)
        
        # default Ternus color
        ternusColor = "black"
        
        # reset every trial
        leftFrame2_Color = "black"
        middleFrame2_Color = "black"
        rightFrame2_Color = "black"
        
        if conditions_practice == "Portal":
            portalOpacity = 1
            barrierOpacity = 0
        
        
        elif conditions_practice == "Barrier":
            portalOpacity = 0
            barrierOpacity = 1
        
        
        elif conditions_practice == "Standard":
            portalOpacity = 0
            barrierOpacity = 0
            ternusColor = "black"
        
        # color-change only in Portal/Barrier
        if conditions_practice == "Portal" or conditions_practice == "Barrier":
            if direction == "LR":
                rightFrame2_Color = causalColor
                barrierPos = (2,2)
            elif direction == "RL":
                leftFrame2_Color = causalColor
                barrierPos = (-2,2)
        
        thisExp.addData("causalColor", causalColor)
        thisExp.addData("direction", direction)
        ternusObjLeftPrac.setPos(leftFrame1_Pos)
        ternusObjMiddlePrac.setPos(middleFrame1_Pos)
        ternusObjRightPrac.setPos(rightFrame1_Pos)
        portalLeftPrac.setFillColor(causalColor)
        portalLeftPrac.setOpacity(portalOpacity)
        portalLeftPrac.setLineColor(causalColor)
        portalRightPrac.setFillColor(causalColor)
        portalRightPrac.setOpacity(portalOpacity)
        portalRightPrac.setLineColor(causalColor)
        BarrierPrac.setFillColor(causalColor)
        BarrierPrac.setOpacity(barrierOpacity)
        BarrierPrac.setPos(barrierPos)
        BarrierPrac.setLineColor(causalColor)
        # store start times for practiceFrame1
        practiceFrame1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practiceFrame1.tStart = globalClock.getTime(format='float')
        practiceFrame1.status = STARTED
        thisExp.addData('practiceFrame1.started', practiceFrame1.tStart)
        practiceFrame1.maxDuration = None
        # keep track of which components have finished
        practiceFrame1Components = practiceFrame1.components
        for thisComponent in practiceFrame1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practiceFrame1" ---
        practiceFrame1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracticeLoop, 'status') and thisPracticeLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ternusObjLeftPrac* updates
            
            # if ternusObjLeftPrac is starting this frame...
            if ternusObjLeftPrac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusObjLeftPrac.frameNStart = frameN  # exact frame index
                ternusObjLeftPrac.tStart = t  # local t and not account for scr refresh
                ternusObjLeftPrac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusObjLeftPrac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusObjLeftPrac.started')
                # update status
                ternusObjLeftPrac.status = STARTED
                ternusObjLeftPrac.setAutoDraw(True)
            
            # if ternusObjLeftPrac is active this frame...
            if ternusObjLeftPrac.status == STARTED:
                # update params
                pass
            
            # if ternusObjLeftPrac is stopping this frame...
            if ternusObjLeftPrac.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusObjLeftPrac.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusObjLeftPrac.tStop = t  # not accounting for scr refresh
                    ternusObjLeftPrac.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusObjLeftPrac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusObjLeftPrac.stopped')
                    # update status
                    ternusObjLeftPrac.status = FINISHED
                    ternusObjLeftPrac.setAutoDraw(False)
            
            # *ternusObjMiddlePrac* updates
            
            # if ternusObjMiddlePrac is starting this frame...
            if ternusObjMiddlePrac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusObjMiddlePrac.frameNStart = frameN  # exact frame index
                ternusObjMiddlePrac.tStart = t  # local t and not account for scr refresh
                ternusObjMiddlePrac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusObjMiddlePrac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusObjMiddlePrac.started')
                # update status
                ternusObjMiddlePrac.status = STARTED
                ternusObjMiddlePrac.setAutoDraw(True)
            
            # if ternusObjMiddlePrac is active this frame...
            if ternusObjMiddlePrac.status == STARTED:
                # update params
                pass
            
            # if ternusObjMiddlePrac is stopping this frame...
            if ternusObjMiddlePrac.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusObjMiddlePrac.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusObjMiddlePrac.tStop = t  # not accounting for scr refresh
                    ternusObjMiddlePrac.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusObjMiddlePrac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusObjMiddlePrac.stopped')
                    # update status
                    ternusObjMiddlePrac.status = FINISHED
                    ternusObjMiddlePrac.setAutoDraw(False)
            
            # *ternusObjRightPrac* updates
            
            # if ternusObjRightPrac is starting this frame...
            if ternusObjRightPrac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusObjRightPrac.frameNStart = frameN  # exact frame index
                ternusObjRightPrac.tStart = t  # local t and not account for scr refresh
                ternusObjRightPrac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusObjRightPrac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusObjRightPrac.started')
                # update status
                ternusObjRightPrac.status = STARTED
                ternusObjRightPrac.setAutoDraw(True)
            
            # if ternusObjRightPrac is active this frame...
            if ternusObjRightPrac.status == STARTED:
                # update params
                pass
            
            # if ternusObjRightPrac is stopping this frame...
            if ternusObjRightPrac.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusObjRightPrac.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusObjRightPrac.tStop = t  # not accounting for scr refresh
                    ternusObjRightPrac.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusObjRightPrac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusObjRightPrac.stopped')
                    # update status
                    ternusObjRightPrac.status = FINISHED
                    ternusObjRightPrac.setAutoDraw(False)
            
            # *FixationCrossPrac* updates
            
            # if FixationCrossPrac is starting this frame...
            if FixationCrossPrac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCrossPrac.frameNStart = frameN  # exact frame index
                FixationCrossPrac.tStart = t  # local t and not account for scr refresh
                FixationCrossPrac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCrossPrac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationCrossPrac.started')
                # update status
                FixationCrossPrac.status = STARTED
                FixationCrossPrac.setAutoDraw(True)
            
            # if FixationCrossPrac is active this frame...
            if FixationCrossPrac.status == STARTED:
                # update params
                pass
            
            # if FixationCrossPrac is stopping this frame...
            if FixationCrossPrac.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCrossPrac.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCrossPrac.tStop = t  # not accounting for scr refresh
                    FixationCrossPrac.tStopRefresh = tThisFlipGlobal  # on global time
                    FixationCrossPrac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCrossPrac.stopped')
                    # update status
                    FixationCrossPrac.status = FINISHED
                    FixationCrossPrac.setAutoDraw(False)
            
            # *portalLeftPrac* updates
            
            # if portalLeftPrac is starting this frame...
            if portalLeftPrac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                portalLeftPrac.frameNStart = frameN  # exact frame index
                portalLeftPrac.tStart = t  # local t and not account for scr refresh
                portalLeftPrac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(portalLeftPrac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'portalLeftPrac.started')
                # update status
                portalLeftPrac.status = STARTED
                portalLeftPrac.setAutoDraw(True)
            
            # if portalLeftPrac is active this frame...
            if portalLeftPrac.status == STARTED:
                # update params
                pass
            
            # if portalLeftPrac is stopping this frame...
            if portalLeftPrac.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > portalLeftPrac.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    portalLeftPrac.tStop = t  # not accounting for scr refresh
                    portalLeftPrac.tStopRefresh = tThisFlipGlobal  # on global time
                    portalLeftPrac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'portalLeftPrac.stopped')
                    # update status
                    portalLeftPrac.status = FINISHED
                    portalLeftPrac.setAutoDraw(False)
            
            # *portalRightPrac* updates
            
            # if portalRightPrac is starting this frame...
            if portalRightPrac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                portalRightPrac.frameNStart = frameN  # exact frame index
                portalRightPrac.tStart = t  # local t and not account for scr refresh
                portalRightPrac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(portalRightPrac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'portalRightPrac.started')
                # update status
                portalRightPrac.status = STARTED
                portalRightPrac.setAutoDraw(True)
            
            # if portalRightPrac is active this frame...
            if portalRightPrac.status == STARTED:
                # update params
                pass
            
            # if portalRightPrac is stopping this frame...
            if portalRightPrac.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > portalRightPrac.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    portalRightPrac.tStop = t  # not accounting for scr refresh
                    portalRightPrac.tStopRefresh = tThisFlipGlobal  # on global time
                    portalRightPrac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'portalRightPrac.stopped')
                    # update status
                    portalRightPrac.status = FINISHED
                    portalRightPrac.setAutoDraw(False)
            
            # *BarrierPrac* updates
            
            # if BarrierPrac is starting this frame...
            if BarrierPrac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                BarrierPrac.frameNStart = frameN  # exact frame index
                BarrierPrac.tStart = t  # local t and not account for scr refresh
                BarrierPrac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(BarrierPrac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'BarrierPrac.started')
                # update status
                BarrierPrac.status = STARTED
                BarrierPrac.setAutoDraw(True)
            
            # if BarrierPrac is active this frame...
            if BarrierPrac.status == STARTED:
                # update params
                pass
            
            # if BarrierPrac is stopping this frame...
            if BarrierPrac.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > BarrierPrac.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    BarrierPrac.tStop = t  # not accounting for scr refresh
                    BarrierPrac.tStopRefresh = tThisFlipGlobal  # on global time
                    BarrierPrac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'BarrierPrac.stopped')
                    # update status
                    BarrierPrac.status = FINISHED
                    BarrierPrac.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=practiceFrame1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                practiceFrame1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practiceFrame1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practiceFrame1" ---
        for thisComponent in practiceFrame1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practiceFrame1
        practiceFrame1.tStop = globalClock.getTime(format='float')
        practiceFrame1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practiceFrame1.stopped', practiceFrame1.tStop)
        # the Routine "practiceFrame1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blankISI_Practice" ---
        # create an object to store info about Routine blankISI_Practice
        blankISI_Practice = data.Routine(
            name='blankISI_Practice',
            components=[FixationCrossBlankPractice],
        )
        blankISI_Practice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blankISI_Practice
        blankISI_Practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blankISI_Practice.tStart = globalClock.getTime(format='float')
        blankISI_Practice.status = STARTED
        thisExp.addData('blankISI_Practice.started', blankISI_Practice.tStart)
        blankISI_Practice.maxDuration = None
        # keep track of which components have finished
        blankISI_PracticeComponents = blankISI_Practice.components
        for thisComponent in blankISI_Practice.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blankISI_Practice" ---
        blankISI_Practice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracticeLoop, 'status') and thisPracticeLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *FixationCrossBlankPractice* updates
            
            # if FixationCrossBlankPractice is starting this frame...
            if FixationCrossBlankPractice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCrossBlankPractice.frameNStart = frameN  # exact frame index
                FixationCrossBlankPractice.tStart = t  # local t and not account for scr refresh
                FixationCrossBlankPractice.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCrossBlankPractice, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationCrossBlankPractice.started')
                # update status
                FixationCrossBlankPractice.status = STARTED
                FixationCrossBlankPractice.setAutoDraw(True)
            
            # if FixationCrossBlankPractice is active this frame...
            if FixationCrossBlankPractice.status == STARTED:
                # update params
                pass
            
            # if FixationCrossBlankPractice is stopping this frame...
            if FixationCrossBlankPractice.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCrossBlankPractice.tStartRefresh + ISI_practice-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCrossBlankPractice.tStop = t  # not accounting for scr refresh
                    FixationCrossBlankPractice.tStopRefresh = tThisFlipGlobal  # on global time
                    FixationCrossBlankPractice.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCrossBlankPractice.stopped')
                    # update status
                    FixationCrossBlankPractice.status = FINISHED
                    FixationCrossBlankPractice.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blankISI_Practice,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blankISI_Practice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blankISI_Practice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blankISI_Practice" ---
        for thisComponent in blankISI_Practice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blankISI_Practice
        blankISI_Practice.tStop = globalClock.getTime(format='float')
        blankISI_Practice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blankISI_Practice.stopped', blankISI_Practice.tStop)
        # the Routine "blankISI_Practice" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "practiceFrame2" ---
        # create an object to store info about Routine practiceFrame2
        practiceFrame2 = data.Routine(
            name='practiceFrame2',
            components=[keyPractice, ternusObjLeftPrac_2, ternusObjMiddlePrac_2, ternusObjRightPrac_2, FixationCrossPrac_2, portalLeftPrac_2, portalRightPrac_2, BarrierPrac_2],
        )
        practiceFrame2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for keyPractice
        keyPractice.keys = []
        keyPractice.rt = []
        _keyPractice_allKeys = []
        ternusObjLeftPrac_2.setFillColor(leftFrame2_Color)
        ternusObjLeftPrac_2.setPos(leftFrame2_Pos)
        ternusObjLeftPrac_2.setLineColor(leftFrame2_Color)
        ternusObjMiddlePrac_2.setPos(middleFrame2_Pos)
        ternusObjRightPrac_2.setFillColor(rightFrame2_Color)
        ternusObjRightPrac_2.setPos(rightFrame2_Pos)
        ternusObjRightPrac_2.setLineColor(rightFrame2_Color)
        portalLeftPrac_2.setFillColor(causalColor)
        portalLeftPrac_2.setOpacity(portalOpacity)
        portalLeftPrac_2.setLineColor(causalColor)
        portalRightPrac_2.setFillColor(causalColor)
        portalRightPrac_2.setOpacity(portalOpacity)
        portalRightPrac_2.setLineColor(causalColor)
        BarrierPrac_2.setFillColor(causalColor)
        BarrierPrac_2.setOpacity(barrierOpacity)
        BarrierPrac_2.setPos(barrierPos)
        BarrierPrac_2.setLineColor(causalColor)
        # store start times for practiceFrame2
        practiceFrame2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practiceFrame2.tStart = globalClock.getTime(format='float')
        practiceFrame2.status = STARTED
        thisExp.addData('practiceFrame2.started', practiceFrame2.tStart)
        practiceFrame2.maxDuration = None
        # keep track of which components have finished
        practiceFrame2Components = practiceFrame2.components
        for thisComponent in practiceFrame2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practiceFrame2" ---
        practiceFrame2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracticeLoop, 'status') and thisPracticeLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *keyPractice* updates
            waitOnFlip = False
            
            # if keyPractice is starting this frame...
            if keyPractice.status == NOT_STARTED and tThisFlip >= ternusDuration-frameTolerance:
                # keep track of start time/frame for later
                keyPractice.frameNStart = frameN  # exact frame index
                keyPractice.tStart = t  # local t and not account for scr refresh
                keyPractice.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(keyPractice, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'keyPractice.started')
                # update status
                keyPractice.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(keyPractice.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(keyPractice.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if keyPractice.status == STARTED and not waitOnFlip:
                theseKeys = keyPractice.getKeys(keyList=['f','j'], ignoreKeys=["escape"], waitRelease=False)
                _keyPractice_allKeys.extend(theseKeys)
                if len(_keyPractice_allKeys):
                    keyPractice.keys = _keyPractice_allKeys[-1].name  # just the last key pressed
                    keyPractice.rt = _keyPractice_allKeys[-1].rt
                    keyPractice.duration = _keyPractice_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *ternusObjLeftPrac_2* updates
            
            # if ternusObjLeftPrac_2 is starting this frame...
            if ternusObjLeftPrac_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusObjLeftPrac_2.frameNStart = frameN  # exact frame index
                ternusObjLeftPrac_2.tStart = t  # local t and not account for scr refresh
                ternusObjLeftPrac_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusObjLeftPrac_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusObjLeftPrac_2.started')
                # update status
                ternusObjLeftPrac_2.status = STARTED
                ternusObjLeftPrac_2.setAutoDraw(True)
            
            # if ternusObjLeftPrac_2 is active this frame...
            if ternusObjLeftPrac_2.status == STARTED:
                # update params
                pass
            
            # if ternusObjLeftPrac_2 is stopping this frame...
            if ternusObjLeftPrac_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusObjLeftPrac_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusObjLeftPrac_2.tStop = t  # not accounting for scr refresh
                    ternusObjLeftPrac_2.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusObjLeftPrac_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusObjLeftPrac_2.stopped')
                    # update status
                    ternusObjLeftPrac_2.status = FINISHED
                    ternusObjLeftPrac_2.setAutoDraw(False)
            
            # *ternusObjMiddlePrac_2* updates
            
            # if ternusObjMiddlePrac_2 is starting this frame...
            if ternusObjMiddlePrac_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusObjMiddlePrac_2.frameNStart = frameN  # exact frame index
                ternusObjMiddlePrac_2.tStart = t  # local t and not account for scr refresh
                ternusObjMiddlePrac_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusObjMiddlePrac_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusObjMiddlePrac_2.started')
                # update status
                ternusObjMiddlePrac_2.status = STARTED
                ternusObjMiddlePrac_2.setAutoDraw(True)
            
            # if ternusObjMiddlePrac_2 is active this frame...
            if ternusObjMiddlePrac_2.status == STARTED:
                # update params
                pass
            
            # if ternusObjMiddlePrac_2 is stopping this frame...
            if ternusObjMiddlePrac_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusObjMiddlePrac_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusObjMiddlePrac_2.tStop = t  # not accounting for scr refresh
                    ternusObjMiddlePrac_2.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusObjMiddlePrac_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusObjMiddlePrac_2.stopped')
                    # update status
                    ternusObjMiddlePrac_2.status = FINISHED
                    ternusObjMiddlePrac_2.setAutoDraw(False)
            
            # *ternusObjRightPrac_2* updates
            
            # if ternusObjRightPrac_2 is starting this frame...
            if ternusObjRightPrac_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ternusObjRightPrac_2.frameNStart = frameN  # exact frame index
                ternusObjRightPrac_2.tStart = t  # local t and not account for scr refresh
                ternusObjRightPrac_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ternusObjRightPrac_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ternusObjRightPrac_2.started')
                # update status
                ternusObjRightPrac_2.status = STARTED
                ternusObjRightPrac_2.setAutoDraw(True)
            
            # if ternusObjRightPrac_2 is active this frame...
            if ternusObjRightPrac_2.status == STARTED:
                # update params
                pass
            
            # if ternusObjRightPrac_2 is stopping this frame...
            if ternusObjRightPrac_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ternusObjRightPrac_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    ternusObjRightPrac_2.tStop = t  # not accounting for scr refresh
                    ternusObjRightPrac_2.tStopRefresh = tThisFlipGlobal  # on global time
                    ternusObjRightPrac_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusObjRightPrac_2.stopped')
                    # update status
                    ternusObjRightPrac_2.status = FINISHED
                    ternusObjRightPrac_2.setAutoDraw(False)
            
            # *FixationCrossPrac_2* updates
            
            # if FixationCrossPrac_2 is starting this frame...
            if FixationCrossPrac_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCrossPrac_2.frameNStart = frameN  # exact frame index
                FixationCrossPrac_2.tStart = t  # local t and not account for scr refresh
                FixationCrossPrac_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCrossPrac_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationCrossPrac_2.started')
                # update status
                FixationCrossPrac_2.status = STARTED
                FixationCrossPrac_2.setAutoDraw(True)
            
            # if FixationCrossPrac_2 is active this frame...
            if FixationCrossPrac_2.status == STARTED:
                # update params
                pass
            
            # *portalLeftPrac_2* updates
            
            # if portalLeftPrac_2 is starting this frame...
            if portalLeftPrac_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                portalLeftPrac_2.frameNStart = frameN  # exact frame index
                portalLeftPrac_2.tStart = t  # local t and not account for scr refresh
                portalLeftPrac_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(portalLeftPrac_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'portalLeftPrac_2.started')
                # update status
                portalLeftPrac_2.status = STARTED
                portalLeftPrac_2.setAutoDraw(True)
            
            # if portalLeftPrac_2 is active this frame...
            if portalLeftPrac_2.status == STARTED:
                # update params
                pass
            
            # if portalLeftPrac_2 is stopping this frame...
            if portalLeftPrac_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > portalLeftPrac_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    portalLeftPrac_2.tStop = t  # not accounting for scr refresh
                    portalLeftPrac_2.tStopRefresh = tThisFlipGlobal  # on global time
                    portalLeftPrac_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'portalLeftPrac_2.stopped')
                    # update status
                    portalLeftPrac_2.status = FINISHED
                    portalLeftPrac_2.setAutoDraw(False)
            
            # *portalRightPrac_2* updates
            
            # if portalRightPrac_2 is starting this frame...
            if portalRightPrac_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                portalRightPrac_2.frameNStart = frameN  # exact frame index
                portalRightPrac_2.tStart = t  # local t and not account for scr refresh
                portalRightPrac_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(portalRightPrac_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'portalRightPrac_2.started')
                # update status
                portalRightPrac_2.status = STARTED
                portalRightPrac_2.setAutoDraw(True)
            
            # if portalRightPrac_2 is active this frame...
            if portalRightPrac_2.status == STARTED:
                # update params
                pass
            
            # if portalRightPrac_2 is stopping this frame...
            if portalRightPrac_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > portalRightPrac_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    portalRightPrac_2.tStop = t  # not accounting for scr refresh
                    portalRightPrac_2.tStopRefresh = tThisFlipGlobal  # on global time
                    portalRightPrac_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'portalRightPrac_2.stopped')
                    # update status
                    portalRightPrac_2.status = FINISHED
                    portalRightPrac_2.setAutoDraw(False)
            
            # *BarrierPrac_2* updates
            
            # if BarrierPrac_2 is starting this frame...
            if BarrierPrac_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                BarrierPrac_2.frameNStart = frameN  # exact frame index
                BarrierPrac_2.tStart = t  # local t and not account for scr refresh
                BarrierPrac_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(BarrierPrac_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'BarrierPrac_2.started')
                # update status
                BarrierPrac_2.status = STARTED
                BarrierPrac_2.setAutoDraw(True)
            
            # if BarrierPrac_2 is active this frame...
            if BarrierPrac_2.status == STARTED:
                # update params
                pass
            
            # if BarrierPrac_2 is stopping this frame...
            if BarrierPrac_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > BarrierPrac_2.tStartRefresh + ternusDuration-frameTolerance:
                    # keep track of stop time/frame for later
                    BarrierPrac_2.tStop = t  # not accounting for scr refresh
                    BarrierPrac_2.tStopRefresh = tThisFlipGlobal  # on global time
                    BarrierPrac_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'BarrierPrac_2.stopped')
                    # update status
                    BarrierPrac_2.status = FINISHED
                    BarrierPrac_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=practiceFrame2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                practiceFrame2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practiceFrame2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practiceFrame2" ---
        for thisComponent in practiceFrame2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practiceFrame2
        practiceFrame2.tStop = globalClock.getTime(format='float')
        practiceFrame2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practiceFrame2.stopped', practiceFrame2.tStop)
        # check responses
        if keyPractice.keys in ['', [], None]:  # No response was made
            keyPractice.keys = None
        practiceLoop.addData('keyPractice.keys',keyPractice.keys)
        if keyPractice.keys != None:  # we had a response
            practiceLoop.addData('keyPractice.rt', keyPractice.rt)
            practiceLoop.addData('keyPractice.duration', keyPractice.duration)
        # the Routine "practiceFrame2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank800" ---
        # create an object to store info about Routine blank800
        blank800 = data.Routine(
            name='blank800',
            components=[FixationCrossBlank800],
        )
        blank800.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank800
        blank800.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank800.tStart = globalClock.getTime(format='float')
        blank800.status = STARTED
        thisExp.addData('blank800.started', blank800.tStart)
        blank800.maxDuration = None
        # keep track of which components have finished
        blank800Components = blank800.components
        for thisComponent in blank800.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank800" ---
        blank800.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.8:
            # if trial has changed, end Routine now
            if hasattr(thisPracticeLoop, 'status') and thisPracticeLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *FixationCrossBlank800* updates
            
            # if FixationCrossBlank800 is starting this frame...
            if FixationCrossBlank800.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCrossBlank800.frameNStart = frameN  # exact frame index
                FixationCrossBlank800.tStart = t  # local t and not account for scr refresh
                FixationCrossBlank800.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCrossBlank800, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationCrossBlank800.started')
                # update status
                FixationCrossBlank800.status = STARTED
                FixationCrossBlank800.setAutoDraw(True)
            
            # if FixationCrossBlank800 is active this frame...
            if FixationCrossBlank800.status == STARTED:
                # update params
                pass
            
            # if FixationCrossBlank800 is stopping this frame...
            if FixationCrossBlank800.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCrossBlank800.tStartRefresh + 0.8-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCrossBlank800.tStop = t  # not accounting for scr refresh
                    FixationCrossBlank800.tStopRefresh = tThisFlipGlobal  # on global time
                    FixationCrossBlank800.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCrossBlank800.stopped')
                    # update status
                    FixationCrossBlank800.status = FINISHED
                    FixationCrossBlank800.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank800,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank800.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank800.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank800" ---
        for thisComponent in blank800.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank800
        blank800.tStop = globalClock.getTime(format='float')
        blank800.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank800.stopped', blank800.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank800.maxDurationReached:
            routineTimer.addTime(-blank800.maxDuration)
        elif blank800.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.800000)
        # mark thisPracticeLoop as finished
        if hasattr(thisPracticeLoop, 'status'):
            thisPracticeLoop.status = FINISHED
        # if awaiting a pause, pause now
        if practiceLoop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            practiceLoop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'practiceLoop'
    practiceLoop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "StartScreen" ---
    # create an object to store info about Routine StartScreen
    StartScreen = data.Routine(
        name='StartScreen',
        components=[keyStart, textStartMessage],
    )
    StartScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for keyStart
    keyStart.keys = []
    keyStart.rt = []
    _keyStart_allKeys = []
    # store start times for StartScreen
    StartScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    StartScreen.tStart = globalClock.getTime(format='float')
    StartScreen.status = STARTED
    thisExp.addData('StartScreen.started', StartScreen.tStart)
    StartScreen.maxDuration = None
    # keep track of which components have finished
    StartScreenComponents = StartScreen.components
    for thisComponent in StartScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "StartScreen" ---
    StartScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *keyStart* updates
        waitOnFlip = False
        
        # if keyStart is starting this frame...
        if keyStart.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            keyStart.frameNStart = frameN  # exact frame index
            keyStart.tStart = t  # local t and not account for scr refresh
            keyStart.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyStart, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyStart.started')
            # update status
            keyStart.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyStart.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyStart.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyStart.status == STARTED and not waitOnFlip:
            theseKeys = keyStart.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _keyStart_allKeys.extend(theseKeys)
            if len(_keyStart_allKeys):
                keyStart.keys = _keyStart_allKeys[-1].name  # just the last key pressed
                keyStart.rt = _keyStart_allKeys[-1].rt
                keyStart.duration = _keyStart_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *textStartMessage* updates
        
        # if textStartMessage is starting this frame...
        if textStartMessage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textStartMessage.frameNStart = frameN  # exact frame index
            textStartMessage.tStart = t  # local t and not account for scr refresh
            textStartMessage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textStartMessage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textStartMessage.started')
            # update status
            textStartMessage.status = STARTED
            textStartMessage.setAutoDraw(True)
        
        # if textStartMessage is active this frame...
        if textStartMessage.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=StartScreen,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            StartScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in StartScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "StartScreen" ---
    for thisComponent in StartScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for StartScreen
    StartScreen.tStop = globalClock.getTime(format='float')
    StartScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('StartScreen.stopped', StartScreen.tStop)
    # check responses
    if keyStart.keys in ['', [], None]:  # No response was made
        keyStart.keys = None
    thisExp.addData('keyStart.keys',keyStart.keys)
    if keyStart.keys != None:  # we had a response
        thisExp.addData('keyStart.rt', keyStart.rt)
        thisExp.addData('keyStart.duration', keyStart.duration)
    thisExp.nextEntry()
    # the Routine "StartScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blockLoop = data.TrialHandler2(
        name='blockLoop',
        nReps=10.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(blockLoop)  # add the loop to the experiment
    thisBlockLoop = blockLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlockLoop.rgb)
    if thisBlockLoop != None:
        for paramName in thisBlockLoop:
            globals()[paramName] = thisBlockLoop[paramName]
    
    for thisBlockLoop in blockLoop:
        blockLoop.status = STARTED
        if hasattr(thisBlockLoop, 'status'):
            thisBlockLoop.status = STARTED
        currentLoop = blockLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisBlockLoop.rgb)
        if thisBlockLoop != None:
            for paramName in thisBlockLoop:
                globals()[paramName] = thisBlockLoop[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trialLoop = data.TrialHandler2(
            name='trialLoop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('trialLoopConditionsTeamprojekt.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(trialLoop)  # add the loop to the experiment
        thisTrialLoop = trialLoop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrialLoop.rgb)
        if thisTrialLoop != None:
            for paramName in thisTrialLoop:
                globals()[paramName] = thisTrialLoop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrialLoop in trialLoop:
            trialLoop.status = STARTED
            if hasattr(thisTrialLoop, 'status'):
                thisTrialLoop.status = STARTED
            currentLoop = trialLoop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrialLoop.rgb)
            if thisTrialLoop != None:
                for paramName in thisTrialLoop:
                    globals()[paramName] = thisTrialLoop[paramName]
            
            # --- Prepare to start Routine "ternusFrame1" ---
            # create an object to store info about Routine ternusFrame1
            ternusFrame1 = data.Routine(
                name='ternusFrame1',
                components=[ternusObjLeft, ternusObjMiddle, ternusObjRight, FixationCross, portalLeft, portalRight, Barrier],
            )
            ternusFrame1.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from codeObjectParameters
            colorList = [
                "#E41A1C",  # red
                "#377EB8",  # blue
                "#4DAF4A",  # green
                "#984EA3",  # purple
                "#FF7F00",  # orange
                "#A65628",  # brown
                "#F781BF",  # pink
                "#1B9E77"   # teal
            ]
            
            causalColor = np.random.choice(colorList)
            
            direction = np.random.choice(["LR", "RL"])
            
            if direction == "LR":
                leftFrame1_Pos = (-3, 2)
                middleFrame1_Pos = (-1, 2)
                rightFrame1_Pos = (1, 2)
            
                leftFrame2_Pos = (-1, 2)
                middleFrame2_Pos = (1, 2)
                rightFrame2_Pos = (3, 2)
            
            elif direction == "RL":
                leftFrame1_Pos = (-1, 2)
                middleFrame1_Pos = (1, 2)
                rightFrame1_Pos = (3, 2)
            
                leftFrame2_Pos = (-3, 2)
                middleFrame2_Pos = (-1, 2)
                rightFrame2_Pos = (1, 2)
            
            # default: invisible causal objects
            portalOpacity = 0
            barrierOpacity = 0
            barrierPos = (2,2)
            
            # default Ternus color
            ternusColor = "black"
            
            # reset every trial
            leftFrame2_Color = "black"
            middleFrame2_Color = "black"
            rightFrame2_Color = "black"
            
            if condition == "Portal":
                portalOpacity = 1
                barrierOpacity = 0
            
            
            elif condition == "Barrier":
                portalOpacity = 0
                barrierOpacity = 1
            
            
            elif condition == "Standard":
                portalOpacity = 0
                barrierOpacity = 0
                ternusColor = "black"
            
            # color-change only in Portal/Barrier
            if condition == "Portal" or condition == "Barrier":
                if direction == "LR":
                    rightFrame2_Color = causalColor
                    barrierPos = (2,2)
                elif direction == "RL":
                    leftFrame2_Color = causalColor
                    barrierPos = (-2,2)
            
            thisExp.addData("causalColor", causalColor)
            thisExp.addData("direction", direction)
            ternusObjLeft.setPos(leftFrame1_Pos)
            ternusObjMiddle.setPos(middleFrame1_Pos)
            ternusObjRight.setPos(rightFrame1_Pos)
            portalLeft.setFillColor(causalColor)
            portalLeft.setOpacity(portalOpacity)
            portalLeft.setLineColor(causalColor)
            portalRight.setFillColor(causalColor)
            portalRight.setOpacity(portalOpacity)
            portalRight.setLineColor(causalColor)
            Barrier.setFillColor(causalColor)
            Barrier.setOpacity(barrierOpacity)
            Barrier.setPos(barrierPos)
            Barrier.setLineColor(causalColor)
            # store start times for ternusFrame1
            ternusFrame1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            ternusFrame1.tStart = globalClock.getTime(format='float')
            ternusFrame1.status = STARTED
            thisExp.addData('ternusFrame1.started', ternusFrame1.tStart)
            ternusFrame1.maxDuration = None
            # keep track of which components have finished
            ternusFrame1Components = ternusFrame1.components
            for thisComponent in ternusFrame1.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "ternusFrame1" ---
            ternusFrame1.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *ternusObjLeft* updates
                
                # if ternusObjLeft is starting this frame...
                if ternusObjLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    ternusObjLeft.frameNStart = frameN  # exact frame index
                    ternusObjLeft.tStart = t  # local t and not account for scr refresh
                    ternusObjLeft.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ternusObjLeft, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusObjLeft.started')
                    # update status
                    ternusObjLeft.status = STARTED
                    ternusObjLeft.setAutoDraw(True)
                
                # if ternusObjLeft is active this frame...
                if ternusObjLeft.status == STARTED:
                    # update params
                    pass
                
                # if ternusObjLeft is stopping this frame...
                if ternusObjLeft.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > ternusObjLeft.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        ternusObjLeft.tStop = t  # not accounting for scr refresh
                        ternusObjLeft.tStopRefresh = tThisFlipGlobal  # on global time
                        ternusObjLeft.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'ternusObjLeft.stopped')
                        # update status
                        ternusObjLeft.status = FINISHED
                        ternusObjLeft.setAutoDraw(False)
                
                # *ternusObjMiddle* updates
                
                # if ternusObjMiddle is starting this frame...
                if ternusObjMiddle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    ternusObjMiddle.frameNStart = frameN  # exact frame index
                    ternusObjMiddle.tStart = t  # local t and not account for scr refresh
                    ternusObjMiddle.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ternusObjMiddle, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusObjMiddle.started')
                    # update status
                    ternusObjMiddle.status = STARTED
                    ternusObjMiddle.setAutoDraw(True)
                
                # if ternusObjMiddle is active this frame...
                if ternusObjMiddle.status == STARTED:
                    # update params
                    pass
                
                # if ternusObjMiddle is stopping this frame...
                if ternusObjMiddle.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > ternusObjMiddle.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        ternusObjMiddle.tStop = t  # not accounting for scr refresh
                        ternusObjMiddle.tStopRefresh = tThisFlipGlobal  # on global time
                        ternusObjMiddle.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'ternusObjMiddle.stopped')
                        # update status
                        ternusObjMiddle.status = FINISHED
                        ternusObjMiddle.setAutoDraw(False)
                
                # *ternusObjRight* updates
                
                # if ternusObjRight is starting this frame...
                if ternusObjRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    ternusObjRight.frameNStart = frameN  # exact frame index
                    ternusObjRight.tStart = t  # local t and not account for scr refresh
                    ternusObjRight.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ternusObjRight, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusObjRight.started')
                    # update status
                    ternusObjRight.status = STARTED
                    ternusObjRight.setAutoDraw(True)
                
                # if ternusObjRight is active this frame...
                if ternusObjRight.status == STARTED:
                    # update params
                    pass
                
                # if ternusObjRight is stopping this frame...
                if ternusObjRight.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > ternusObjRight.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        ternusObjRight.tStop = t  # not accounting for scr refresh
                        ternusObjRight.tStopRefresh = tThisFlipGlobal  # on global time
                        ternusObjRight.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'ternusObjRight.stopped')
                        # update status
                        ternusObjRight.status = FINISHED
                        ternusObjRight.setAutoDraw(False)
                
                # *FixationCross* updates
                
                # if FixationCross is starting this frame...
                if FixationCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    FixationCross.frameNStart = frameN  # exact frame index
                    FixationCross.tStart = t  # local t and not account for scr refresh
                    FixationCross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(FixationCross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCross.started')
                    # update status
                    FixationCross.status = STARTED
                    FixationCross.setAutoDraw(True)
                
                # if FixationCross is active this frame...
                if FixationCross.status == STARTED:
                    # update params
                    pass
                
                # if FixationCross is stopping this frame...
                if FixationCross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > FixationCross.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        FixationCross.tStop = t  # not accounting for scr refresh
                        FixationCross.tStopRefresh = tThisFlipGlobal  # on global time
                        FixationCross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'FixationCross.stopped')
                        # update status
                        FixationCross.status = FINISHED
                        FixationCross.setAutoDraw(False)
                
                # *portalLeft* updates
                
                # if portalLeft is starting this frame...
                if portalLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    portalLeft.frameNStart = frameN  # exact frame index
                    portalLeft.tStart = t  # local t and not account for scr refresh
                    portalLeft.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(portalLeft, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'portalLeft.started')
                    # update status
                    portalLeft.status = STARTED
                    portalLeft.setAutoDraw(True)
                
                # if portalLeft is active this frame...
                if portalLeft.status == STARTED:
                    # update params
                    pass
                
                # if portalLeft is stopping this frame...
                if portalLeft.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > portalLeft.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        portalLeft.tStop = t  # not accounting for scr refresh
                        portalLeft.tStopRefresh = tThisFlipGlobal  # on global time
                        portalLeft.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'portalLeft.stopped')
                        # update status
                        portalLeft.status = FINISHED
                        portalLeft.setAutoDraw(False)
                
                # *portalRight* updates
                
                # if portalRight is starting this frame...
                if portalRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    portalRight.frameNStart = frameN  # exact frame index
                    portalRight.tStart = t  # local t and not account for scr refresh
                    portalRight.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(portalRight, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'portalRight.started')
                    # update status
                    portalRight.status = STARTED
                    portalRight.setAutoDraw(True)
                
                # if portalRight is active this frame...
                if portalRight.status == STARTED:
                    # update params
                    pass
                
                # if portalRight is stopping this frame...
                if portalRight.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > portalRight.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        portalRight.tStop = t  # not accounting for scr refresh
                        portalRight.tStopRefresh = tThisFlipGlobal  # on global time
                        portalRight.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'portalRight.stopped')
                        # update status
                        portalRight.status = FINISHED
                        portalRight.setAutoDraw(False)
                
                # *Barrier* updates
                
                # if Barrier is starting this frame...
                if Barrier.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Barrier.frameNStart = frameN  # exact frame index
                    Barrier.tStart = t  # local t and not account for scr refresh
                    Barrier.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Barrier, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Barrier.started')
                    # update status
                    Barrier.status = STARTED
                    Barrier.setAutoDraw(True)
                
                # if Barrier is active this frame...
                if Barrier.status == STARTED:
                    # update params
                    pass
                
                # if Barrier is stopping this frame...
                if Barrier.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Barrier.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        Barrier.tStop = t  # not accounting for scr refresh
                        Barrier.tStopRefresh = tThisFlipGlobal  # on global time
                        Barrier.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Barrier.stopped')
                        # update status
                        Barrier.status = FINISHED
                        Barrier.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=ternusFrame1,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    ternusFrame1.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ternusFrame1.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "ternusFrame1" ---
            for thisComponent in ternusFrame1.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for ternusFrame1
            ternusFrame1.tStop = globalClock.getTime(format='float')
            ternusFrame1.tStopRefresh = tThisFlipGlobal
            thisExp.addData('ternusFrame1.stopped', ternusFrame1.tStop)
            # the Routine "ternusFrame1" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "blankISI" ---
            # create an object to store info about Routine blankISI
            blankISI = data.Routine(
                name='blankISI',
                components=[FixationCrossBlank],
            )
            blankISI.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for blankISI
            blankISI.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            blankISI.tStart = globalClock.getTime(format='float')
            blankISI.status = STARTED
            thisExp.addData('blankISI.started', blankISI.tStart)
            blankISI.maxDuration = None
            # keep track of which components have finished
            blankISIComponents = blankISI.components
            for thisComponent in blankISI.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "blankISI" ---
            blankISI.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *FixationCrossBlank* updates
                
                # if FixationCrossBlank is starting this frame...
                if FixationCrossBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    FixationCrossBlank.frameNStart = frameN  # exact frame index
                    FixationCrossBlank.tStart = t  # local t and not account for scr refresh
                    FixationCrossBlank.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(FixationCrossBlank, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCrossBlank.started')
                    # update status
                    FixationCrossBlank.status = STARTED
                    FixationCrossBlank.setAutoDraw(True)
                
                # if FixationCrossBlank is active this frame...
                if FixationCrossBlank.status == STARTED:
                    # update params
                    pass
                
                # if FixationCrossBlank is stopping this frame...
                if FixationCrossBlank.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > FixationCrossBlank.tStartRefresh + interstim_interval-frameTolerance:
                        # keep track of stop time/frame for later
                        FixationCrossBlank.tStop = t  # not accounting for scr refresh
                        FixationCrossBlank.tStopRefresh = tThisFlipGlobal  # on global time
                        FixationCrossBlank.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'FixationCrossBlank.stopped')
                        # update status
                        FixationCrossBlank.status = FINISHED
                        FixationCrossBlank.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=blankISI,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    blankISI.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blankISI.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blankISI" ---
            for thisComponent in blankISI.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for blankISI
            blankISI.tStop = globalClock.getTime(format='float')
            blankISI.tStopRefresh = tThisFlipGlobal
            thisExp.addData('blankISI.stopped', blankISI.tStop)
            # the Routine "blankISI" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "ternusFrame2" ---
            # create an object to store info about Routine ternusFrame2
            ternusFrame2 = data.Routine(
                name='ternusFrame2',
                components=[key_resp, ternusObjLeft_2, ternusObjMiddle_2, ternusObjRight_2, FixationCross_2, portalLeft_2, portalRight_2, Barrier_2],
            )
            ternusFrame2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            ternusObjLeft_2.setFillColor(leftFrame2_Color)
            ternusObjLeft_2.setPos(leftFrame2_Pos)
            ternusObjLeft_2.setLineColor(leftFrame2_Color)
            ternusObjMiddle_2.setPos(middleFrame2_Pos)
            ternusObjRight_2.setFillColor(rightFrame2_Color)
            ternusObjRight_2.setPos(rightFrame2_Pos)
            ternusObjRight_2.setLineColor(rightFrame2_Color)
            portalLeft_2.setFillColor(causalColor)
            portalLeft_2.setOpacity(portalOpacity)
            portalLeft_2.setLineColor(causalColor)
            portalRight_2.setFillColor(causalColor)
            portalRight_2.setOpacity(portalOpacity)
            portalRight_2.setLineColor(causalColor)
            Barrier_2.setFillColor(causalColor)
            Barrier_2.setOpacity(barrierOpacity)
            Barrier_2.setPos(barrierPos)
            Barrier_2.setLineColor(causalColor)
            # store start times for ternusFrame2
            ternusFrame2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            ternusFrame2.tStart = globalClock.getTime(format='float')
            ternusFrame2.status = STARTED
            thisExp.addData('ternusFrame2.started', ternusFrame2.tStart)
            ternusFrame2.maxDuration = None
            # keep track of which components have finished
            ternusFrame2Components = ternusFrame2.components
            for thisComponent in ternusFrame2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "ternusFrame2" ---
            ternusFrame2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= ternusDuration-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['f','j'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                        key_resp.rt = _key_resp_allKeys[-1].rt
                        key_resp.duration = _key_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *ternusObjLeft_2* updates
                
                # if ternusObjLeft_2 is starting this frame...
                if ternusObjLeft_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    ternusObjLeft_2.frameNStart = frameN  # exact frame index
                    ternusObjLeft_2.tStart = t  # local t and not account for scr refresh
                    ternusObjLeft_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ternusObjLeft_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusObjLeft_2.started')
                    # update status
                    ternusObjLeft_2.status = STARTED
                    ternusObjLeft_2.setAutoDraw(True)
                
                # if ternusObjLeft_2 is active this frame...
                if ternusObjLeft_2.status == STARTED:
                    # update params
                    pass
                
                # if ternusObjLeft_2 is stopping this frame...
                if ternusObjLeft_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > ternusObjLeft_2.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        ternusObjLeft_2.tStop = t  # not accounting for scr refresh
                        ternusObjLeft_2.tStopRefresh = tThisFlipGlobal  # on global time
                        ternusObjLeft_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'ternusObjLeft_2.stopped')
                        # update status
                        ternusObjLeft_2.status = FINISHED
                        ternusObjLeft_2.setAutoDraw(False)
                
                # *ternusObjMiddle_2* updates
                
                # if ternusObjMiddle_2 is starting this frame...
                if ternusObjMiddle_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    ternusObjMiddle_2.frameNStart = frameN  # exact frame index
                    ternusObjMiddle_2.tStart = t  # local t and not account for scr refresh
                    ternusObjMiddle_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ternusObjMiddle_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusObjMiddle_2.started')
                    # update status
                    ternusObjMiddle_2.status = STARTED
                    ternusObjMiddle_2.setAutoDraw(True)
                
                # if ternusObjMiddle_2 is active this frame...
                if ternusObjMiddle_2.status == STARTED:
                    # update params
                    pass
                
                # if ternusObjMiddle_2 is stopping this frame...
                if ternusObjMiddle_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > ternusObjMiddle_2.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        ternusObjMiddle_2.tStop = t  # not accounting for scr refresh
                        ternusObjMiddle_2.tStopRefresh = tThisFlipGlobal  # on global time
                        ternusObjMiddle_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'ternusObjMiddle_2.stopped')
                        # update status
                        ternusObjMiddle_2.status = FINISHED
                        ternusObjMiddle_2.setAutoDraw(False)
                
                # *ternusObjRight_2* updates
                
                # if ternusObjRight_2 is starting this frame...
                if ternusObjRight_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    ternusObjRight_2.frameNStart = frameN  # exact frame index
                    ternusObjRight_2.tStart = t  # local t and not account for scr refresh
                    ternusObjRight_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ternusObjRight_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ternusObjRight_2.started')
                    # update status
                    ternusObjRight_2.status = STARTED
                    ternusObjRight_2.setAutoDraw(True)
                
                # if ternusObjRight_2 is active this frame...
                if ternusObjRight_2.status == STARTED:
                    # update params
                    pass
                
                # if ternusObjRight_2 is stopping this frame...
                if ternusObjRight_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > ternusObjRight_2.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        ternusObjRight_2.tStop = t  # not accounting for scr refresh
                        ternusObjRight_2.tStopRefresh = tThisFlipGlobal  # on global time
                        ternusObjRight_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'ternusObjRight_2.stopped')
                        # update status
                        ternusObjRight_2.status = FINISHED
                        ternusObjRight_2.setAutoDraw(False)
                
                # *FixationCross_2* updates
                
                # if FixationCross_2 is starting this frame...
                if FixationCross_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    FixationCross_2.frameNStart = frameN  # exact frame index
                    FixationCross_2.tStart = t  # local t and not account for scr refresh
                    FixationCross_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(FixationCross_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCross_2.started')
                    # update status
                    FixationCross_2.status = STARTED
                    FixationCross_2.setAutoDraw(True)
                
                # if FixationCross_2 is active this frame...
                if FixationCross_2.status == STARTED:
                    # update params
                    pass
                
                # *portalLeft_2* updates
                
                # if portalLeft_2 is starting this frame...
                if portalLeft_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    portalLeft_2.frameNStart = frameN  # exact frame index
                    portalLeft_2.tStart = t  # local t and not account for scr refresh
                    portalLeft_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(portalLeft_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'portalLeft_2.started')
                    # update status
                    portalLeft_2.status = STARTED
                    portalLeft_2.setAutoDraw(True)
                
                # if portalLeft_2 is active this frame...
                if portalLeft_2.status == STARTED:
                    # update params
                    pass
                
                # if portalLeft_2 is stopping this frame...
                if portalLeft_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > portalLeft_2.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        portalLeft_2.tStop = t  # not accounting for scr refresh
                        portalLeft_2.tStopRefresh = tThisFlipGlobal  # on global time
                        portalLeft_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'portalLeft_2.stopped')
                        # update status
                        portalLeft_2.status = FINISHED
                        portalLeft_2.setAutoDraw(False)
                
                # *portalRight_2* updates
                
                # if portalRight_2 is starting this frame...
                if portalRight_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    portalRight_2.frameNStart = frameN  # exact frame index
                    portalRight_2.tStart = t  # local t and not account for scr refresh
                    portalRight_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(portalRight_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'portalRight_2.started')
                    # update status
                    portalRight_2.status = STARTED
                    portalRight_2.setAutoDraw(True)
                
                # if portalRight_2 is active this frame...
                if portalRight_2.status == STARTED:
                    # update params
                    pass
                
                # if portalRight_2 is stopping this frame...
                if portalRight_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > portalRight_2.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        portalRight_2.tStop = t  # not accounting for scr refresh
                        portalRight_2.tStopRefresh = tThisFlipGlobal  # on global time
                        portalRight_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'portalRight_2.stopped')
                        # update status
                        portalRight_2.status = FINISHED
                        portalRight_2.setAutoDraw(False)
                
                # *Barrier_2* updates
                
                # if Barrier_2 is starting this frame...
                if Barrier_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Barrier_2.frameNStart = frameN  # exact frame index
                    Barrier_2.tStart = t  # local t and not account for scr refresh
                    Barrier_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Barrier_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Barrier_2.started')
                    # update status
                    Barrier_2.status = STARTED
                    Barrier_2.setAutoDraw(True)
                
                # if Barrier_2 is active this frame...
                if Barrier_2.status == STARTED:
                    # update params
                    pass
                
                # if Barrier_2 is stopping this frame...
                if Barrier_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Barrier_2.tStartRefresh + ternusDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        Barrier_2.tStop = t  # not accounting for scr refresh
                        Barrier_2.tStopRefresh = tThisFlipGlobal  # on global time
                        Barrier_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Barrier_2.stopped')
                        # update status
                        Barrier_2.status = FINISHED
                        Barrier_2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=ternusFrame2,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    ternusFrame2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ternusFrame2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "ternusFrame2" ---
            for thisComponent in ternusFrame2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for ternusFrame2
            ternusFrame2.tStop = globalClock.getTime(format='float')
            ternusFrame2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('ternusFrame2.stopped', ternusFrame2.tStop)
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
            trialLoop.addData('key_resp.keys',key_resp.keys)
            if key_resp.keys != None:  # we had a response
                trialLoop.addData('key_resp.rt', key_resp.rt)
                trialLoop.addData('key_resp.duration', key_resp.duration)
            # the Routine "ternusFrame2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "blank800" ---
            # create an object to store info about Routine blank800
            blank800 = data.Routine(
                name='blank800',
                components=[FixationCrossBlank800],
            )
            blank800.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for blank800
            blank800.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            blank800.tStart = globalClock.getTime(format='float')
            blank800.status = STARTED
            thisExp.addData('blank800.started', blank800.tStart)
            blank800.maxDuration = None
            # keep track of which components have finished
            blank800Components = blank800.components
            for thisComponent in blank800.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "blank800" ---
            blank800.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.8:
                # if trial has changed, end Routine now
                if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *FixationCrossBlank800* updates
                
                # if FixationCrossBlank800 is starting this frame...
                if FixationCrossBlank800.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    FixationCrossBlank800.frameNStart = frameN  # exact frame index
                    FixationCrossBlank800.tStart = t  # local t and not account for scr refresh
                    FixationCrossBlank800.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(FixationCrossBlank800, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationCrossBlank800.started')
                    # update status
                    FixationCrossBlank800.status = STARTED
                    FixationCrossBlank800.setAutoDraw(True)
                
                # if FixationCrossBlank800 is active this frame...
                if FixationCrossBlank800.status == STARTED:
                    # update params
                    pass
                
                # if FixationCrossBlank800 is stopping this frame...
                if FixationCrossBlank800.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > FixationCrossBlank800.tStartRefresh + 0.8-frameTolerance:
                        # keep track of stop time/frame for later
                        FixationCrossBlank800.tStop = t  # not accounting for scr refresh
                        FixationCrossBlank800.tStopRefresh = tThisFlipGlobal  # on global time
                        FixationCrossBlank800.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'FixationCrossBlank800.stopped')
                        # update status
                        FixationCrossBlank800.status = FINISHED
                        FixationCrossBlank800.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=blank800,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    blank800.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blank800.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank800" ---
            for thisComponent in blank800.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for blank800
            blank800.tStop = globalClock.getTime(format='float')
            blank800.tStopRefresh = tThisFlipGlobal
            thisExp.addData('blank800.stopped', blank800.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if blank800.maxDurationReached:
                routineTimer.addTime(-blank800.maxDuration)
            elif blank800.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.800000)
            # mark thisTrialLoop as finished
            if hasattr(thisTrialLoop, 'status'):
                thisTrialLoop.status = FINISHED
            # if awaiting a pause, pause now
            if trialLoop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                trialLoop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trialLoop'
        trialLoop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "blockBreak" ---
        # create an object to store info about Routine blockBreak
        blockBreak = data.Routine(
            name='blockBreak',
            components=[keyBlockContinue, textBlockInstructions],
        )
        blockBreak.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeBlockCount
        blockNumber += 1
        
        breakText = (
            "Sehr gut! Du hast Block " +str(blockNumber) + " von 10 abgeschlossen.\n\n"
            "Mach gerne eine kurze Pause.\n\n"
            "Bitte drücke die 'F'-Taste, wenn sich die Elemente unabhängig voneinander bewegen "
            "(Elementbewegung), und die 'J'-Taste, wenn sich die Elemente gemeinsam "
            " bewegen (Gruppenbewegung).\n\n"
            "Drücke die Leertaste, um mit dem nächsten Block fortzufahren."
        )
        # create starting attributes for keyBlockContinue
        keyBlockContinue.keys = []
        keyBlockContinue.rt = []
        _keyBlockContinue_allKeys = []
        textBlockInstructions.setText(breakText)
        # store start times for blockBreak
        blockBreak.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blockBreak.tStart = globalClock.getTime(format='float')
        blockBreak.status = STARTED
        thisExp.addData('blockBreak.started', blockBreak.tStart)
        blockBreak.maxDuration = None
        # keep track of which components have finished
        blockBreakComponents = blockBreak.components
        for thisComponent in blockBreak.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blockBreak" ---
        blockBreak.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlockLoop, 'status') and thisBlockLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *keyBlockContinue* updates
            waitOnFlip = False
            
            # if keyBlockContinue is starting this frame...
            if keyBlockContinue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                keyBlockContinue.frameNStart = frameN  # exact frame index
                keyBlockContinue.tStart = t  # local t and not account for scr refresh
                keyBlockContinue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(keyBlockContinue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'keyBlockContinue.started')
                # update status
                keyBlockContinue.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(keyBlockContinue.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(keyBlockContinue.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if keyBlockContinue.status == STARTED and not waitOnFlip:
                theseKeys = keyBlockContinue.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _keyBlockContinue_allKeys.extend(theseKeys)
                if len(_keyBlockContinue_allKeys):
                    keyBlockContinue.keys = _keyBlockContinue_allKeys[-1].name  # just the last key pressed
                    keyBlockContinue.rt = _keyBlockContinue_allKeys[-1].rt
                    keyBlockContinue.duration = _keyBlockContinue_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *textBlockInstructions* updates
            
            # if textBlockInstructions is starting this frame...
            if textBlockInstructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textBlockInstructions.frameNStart = frameN  # exact frame index
                textBlockInstructions.tStart = t  # local t and not account for scr refresh
                textBlockInstructions.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textBlockInstructions, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlockInstructions.started')
                # update status
                textBlockInstructions.status = STARTED
                textBlockInstructions.setAutoDraw(True)
            
            # if textBlockInstructions is active this frame...
            if textBlockInstructions.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blockBreak,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blockBreak.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blockBreak.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blockBreak" ---
        for thisComponent in blockBreak.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blockBreak
        blockBreak.tStop = globalClock.getTime(format='float')
        blockBreak.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blockBreak.stopped', blockBreak.tStop)
        # check responses
        if keyBlockContinue.keys in ['', [], None]:  # No response was made
            keyBlockContinue.keys = None
        blockLoop.addData('keyBlockContinue.keys',keyBlockContinue.keys)
        if keyBlockContinue.keys != None:  # we had a response
            blockLoop.addData('keyBlockContinue.rt', keyBlockContinue.rt)
            blockLoop.addData('keyBlockContinue.duration', keyBlockContinue.duration)
        # the Routine "blockBreak" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisBlockLoop as finished
        if hasattr(thisBlockLoop, 'status'):
            thisBlockLoop.status = FINISHED
        # if awaiting a pause, pause now
        if blockLoop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            blockLoop.status = STARTED
    # completed 10.0 repeats of 'blockLoop'
    blockLoop.status = FINISHED
    
    
    # --- Prepare to start Routine "EndScreen" ---
    # create an object to store info about Routine EndScreen
    EndScreen = data.Routine(
        name='EndScreen',
        components=[textEnd],
    )
    EndScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for EndScreen
    EndScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    EndScreen.tStart = globalClock.getTime(format='float')
    EndScreen.status = STARTED
    thisExp.addData('EndScreen.started', EndScreen.tStart)
    EndScreen.maxDuration = None
    # keep track of which components have finished
    EndScreenComponents = EndScreen.components
    for thisComponent in EndScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "EndScreen" ---
    EndScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textEnd* updates
        
        # if textEnd is starting this frame...
        if textEnd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textEnd.frameNStart = frameN  # exact frame index
            textEnd.tStart = t  # local t and not account for scr refresh
            textEnd.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textEnd, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textEnd.started')
            # update status
            textEnd.status = STARTED
            textEnd.setAutoDraw(True)
        
        # if textEnd is active this frame...
        if textEnd.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=EndScreen,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            EndScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in EndScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "EndScreen" ---
    for thisComponent in EndScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for EndScreen
    EndScreen.tStop = globalClock.getTime(format='float')
    EndScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('EndScreen.stopped', EndScreen.tStop)
    thisExp.nextEntry()
    # the Routine "EndScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
