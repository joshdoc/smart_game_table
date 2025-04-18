# Smart Game Table
EECS 498-008 & 598-008 / CSE 596 - Engineering Interactive Systems
## Authors
- Daniel Calco (dcalco@umich.edu) // dcalkoj.github.io
- Joshua Doctor (joshdoc@umich.edu)
- Matthew Priskorn (mcprisk@umich.edu)
- Neel Vora (neelnv@umich.edu)
## Usage
### User Testing
- `user_test.py <DURATION (s)> <NAME> <RADIUS (px)> <MOUSE (0|1)>` \
Runs a single instance of our user test with the specified parameters.
- `user_test.bat <NAME>` \
Runs our full user testing suite for a single user.
- `plot.py` \
Plots all of our data from the user test data.
### Games
- `mouse.py` \
Makes the table into a functional touchscreen.
- Other games... \
TBD - see `game/hockey` and `game/undertable` branches for current status.
### Backend
- `cv.py` \
Run on its own to see detected centroids.  This program is used by the games and
user tests to detect finger presses and tangible object locations.
