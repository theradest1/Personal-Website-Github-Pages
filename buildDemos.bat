:: run this script to build the demos with pygbag
:: (with './buildDemos.bat' in the terminal)

pygbag --ume_block 0 --can_close 1 --build python-projects/Liquid-Sim/main.py
pygbag --ume_block 0 --can_close 1 --build python-projects/gravity-simulator/main.py

:: ume_block 0 makes it so the user doesn't have to click something to start it
:: can_close 1 lets the page close without a confirmation message
:: build makes it so it doesnt run a server after building (a bit annoying)