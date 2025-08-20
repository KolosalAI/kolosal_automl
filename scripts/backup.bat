@echo off
REM Backup script for Kolosal AutoML
set DATE=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set DATE=%DATE: =0%
echo Creating backup at %date% %time%
mkdir backups\%DATE% 2>nul
xcopy models backups\%DATE%\models\ /E /I /Q 2>nul
xcopy logs backups\%DATE%\logs\ /E /I /Q 2>nul
copy .env backups\%DATE%\ 2>nul
powershell Compress-Archive -Path backups\%DATE% -DestinationPath backups\backup_%DATE%.zip
rmdir /S /Q backups\%DATE% 2>nul
echo Backup completed: backup_%DATE%.zip
