import sys
import os
import inspect
import logging
from logging import handlers


# ---------------------------------------------------------------------------------------
# Creat new logger levels

DEBUG_LEVELV_NUM = 6
logging.addLevelName(DEBUG_LEVELV_NUM, "DEBUGV")
def debugv(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(DEBUG_LEVELV_NUM):
        self._log(DEBUG_LEVELV_NUM, message, args, **kws)

logging.Logger.debugv = debugv

DEBUG_LEVELX_NUM = 8
logging.addLevelName(DEBUG_LEVELX_NUM, "DEBUGX")
def debugx(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(DEBUG_LEVELX_NUM):
        self._log(DEBUG_LEVELX_NUM, message, args, **kws)

logging.Logger.debugx = debugx

DEBUG_LEVELW_NUM = 12
logging.addLevelName(DEBUG_LEVELW_NUM, "INFOW")
def infow(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(DEBUG_LEVELW_NUM):
        self._log(DEBUG_LEVELW_NUM, message, args, **kws)

logging.Logger.infow = infow
# ---------------------------------------------------------------------------------------


log = None
logcsv = None


def init_logger(logger=None, name='logger_util'):
    global log
    if logger is not None:
        log = logger
    else:
        import logging
        log = logging.getLogger(name)

def init_csvlogger(logger=None, name='logger_util'):
    global logcsv
    if logger is not None:
        logcsv = logger
    else:
        import logging
        logcsv = logging.getLogger(name)

global logs_fully_setup
logs_fully_setup = None

def get_logger(
    name=None, applog="pegasus1.log", basiclog="pegasus_B.log", advlog="pegasus_ADV.log", level=logging.DEBUG
):
    default = "__app__"
    # formatter = logging.Formatter('%(thread)d : %(levelname)s: %(asctime)s %(funcName)s(%(lineno)d) -- %(message)s',
    formatter = logging.Formatter(
        "%(threadName)s: %(levelname)s: %(asctime)s %(funcName)s(%(lineno)d) -- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_map = {"radio_switch": applog, "__app__": applog, "__basic_log__": basiclog, "__advance_log__": advlog}
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(default)
    # fh = logging.FileHandler(log_map[name])
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    consoleHandler = logging.StreamHandler(sys.stdout)
    logger.addHandler(consoleHandler)
    logger.setLevel(level)
    return logger


def setup_sockt_logger(thelogger, logger_ip):
    socketHandler = logging.handlers.SocketHandler(logger_ip, logging.handlers.DEFAULT_TCP_LOGGING_PORT)
    # don't bother with a formatter, since a socket handler sends the event as
    # an unformatted pickle
    thelogger.addHandler(socketHandler)


def setup_logger(name, fname, level=logging.INFO, mode=''):
    home = os.path.expanduser("~")

    currentFile = __file__  # May be 'my_script', or './my_script' or
    if fname == "":
        fname = os.path.splitext(os.path.basename(currentFile))[0]

    logfname = fname
    # '/home/user/test/my_script.py' depending on exactly how
    # the script was run/loaded.
    realPath = os.path.realpath(currentFile)  # /home/user/test/my_script.py
    dirPath = os.path.dirname(realPath)  # /home/user/test
    dirName = os.path.basename(dirPath)  # test

    if "core" in mode:
        log_dir = "/tmp/x_task_logs/logs/"
        log_bin_dir = "/tmp/x_task_logs/bin/"
    else:
        log_dir = dirPath + "/x_task_logs/logs/"
        log_bin_dir = dirPath + "/x_task_logs/bin/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_bin_dir):
        os.makedirs(log_bin_dir)

    print("LOGGING TO -> {}".format(log_dir, logfname))

    """Function setup as many loggers as you want"""
    formatter = logging.Formatter("%(message)s")
    handler = logging.FileHandler(log_dir + logfname + ".log")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # logger.addHandler(handler)

    return logger


def setuplogs(fname="", name="__app__", level=logging.INFO, socket_logger_ip=None, force_setup=False, mode=''):
    global logs_fully_setup, log_dir, log_bin_dir
    if not force_setup and logs_fully_setup:
        return logging.getLogger(name)

    home = os.path.expanduser("~")

    currentFile = __file__  # May be 'my_script', or './my_script' or
    if fname == "":
        fname = os.path.splitext(os.path.basename(currentFile))[0]

    logfname = fname
    # '/home/user/test/my_script.py' depending on exactly how
    # the script was run/loaded.
    realPath = os.path.realpath(currentFile)  # /home/user/test/my_script.py
    dirPath = os.path.dirname(realPath)  # /home/user/test
    dirName = os.path.basename(dirPath)  # test

    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print("Console LOGGING enabled")
    thelogger = get_logger(name=name, applog=log_dir + logfname + ".log", level=level)

    # if socket_logger_ip is not None:
    #     setup_sockt_logger(thelogger, socket_logger_ip)

    logs_fully_setup = True
    #thelogger.info(f"{name} started, logging to file: " + logfname + ", path: " + log_dir)
    return thelogger

