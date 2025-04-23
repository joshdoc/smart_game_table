# Smart Game Table
EECS 498-008 & 598-008 / CSE 596 - Engineering Interactive Systems
## Authors
- Daniel Calco (dcalco@umich.edu) // [dcalkoj.github.io](https://dcalkoj.github.io)
- Joshua Doctor (joshdoc@umich.edu)
- Matthew Priskorn (mcprisk@umich.edu)
- Neel Vora (neelnv@umich.edu)
## Usage
### User Testing
- `user_test.py <DURATION (s)> <NAME> <RADIUS (px)> <MOUSE (0|1)>` \
Runs a single instance of our user test with the specified parameters.
- `user_test.bat <NAME>` \
Runs our full user testing suite for a single user.
- `log/plotter.py` \
Plots all of our data from the user test data.
### Games
- `sgt.py` \
Runs the full suite of games, starting with the game selection menu.
- `dots.py` \
Runs the user testing game without any of the data collection.
- `hockey.py` \
Makes the table into an air hockey emulator detecting our tangible air hockey paddles.
- `macrodata.py` \
Plays a game similar to the macrodata refinement job in Severance.
- `menu.py` \
Runs the menu for switching between games.
- `mouse.py` \
Makes the table into a functional touchscreen.
- `undertable.py` \
Plays a game similar to an Undertale boss fight on the table.
### Backend
- `cv.py` \
Run on its own to see detected centroids.  This program is used by the games and
user tests to detect finger presses and tangible object locations.
- `sgt_types.py` \
Contains all data types shared between multiple files.
- `debug/` \
Provides a script to determine the proper camera orientation and table outline thresholds.
- `graphics/` \
Contains all of the required graphics for the games.
