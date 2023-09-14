@echo off
REM Remove the REM from the next line to stay up-to-date
REM git pull
REM set COMMANDLINE_ARGS=-smallmodels -autolaunch -forcecpu
set COMMANDLINE_ARGS=-autolaunch
python webui.py %COMMANDLINE_ARGS%
pause
