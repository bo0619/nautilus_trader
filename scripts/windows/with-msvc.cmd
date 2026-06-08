@echo off
setlocal

call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%

set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.rustup\toolchains\1.95.0-x86_64-pc-windows-msvc\bin;%PATH%"

%*
