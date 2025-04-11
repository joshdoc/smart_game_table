@echo off
setlocal enabledelayedexpansion

:: Check if username was provided
if "%~1"=="" (
    echo Usage: user_test.bat <USERNAME>
    exit /b
)

:: Store the username from the command-line argument
set username=%~1'
set time=5

:: Define test parameters: duration radius mouse_mode
set test1=!time! 10 1
set test2=!time! 15 1
set test3=!time! 20 1
set test4=!time! 25 1
set test5=!time! 30 1

set test6=!time! 10 0
set test7=!time! 15 0
set test8=!time! 20 0
set test9=!time! 25 0
set test10=!time! 30 0

set test11=!time! 10 0
set test12=!time! 15 0
set test13=!time! 20 0
set test14=!time! 25 0
set test15=!time! 30 0


:: Loop through tests
for %%T in (1 2 3 4 5 6 7 8 9 10 11 12 13 14 15) do (
    call set test=%%test%%T%%
    for /f "tokens=1,2,3" %%a in ("!test!") do (
        echo Running test: Duration=%%a Radius=%%b MouseMode=%%c
        python user_test.py %%a !username! %%b %%c
        echo -----------------------------
    )
)

echo All tests completed!
pause