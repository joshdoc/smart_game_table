
# Smart Game Table

EECS 498-008 & 598-008 / CSE 596 - Engineering Interactive Systems

![smart game table!](https://github.com/joshdoc/smart_game_table/blob/master/Media/pictures/IMG_9220.jpg?raw=true)

## Authors

- Daniel Calco (dcalco@umich.edu) // [dcalkoj.github.io](https://dcalkoj.github.io)

- Joshua Doctor (joshdoc@umich.edu)

- Matthew Priskorn (mcprisk@umich.edu)

- Neel Vora (neelnv@umich.edu)

## Usage

### User Testing

-  `user_test.py <DURATION (s)> <NAME> <RADIUS (px)> <MOUSE (0|1)>` \
Runs a single instance of our user test with the specified parameters.

-  `user_test.bat <NAME>` \
Runs our full user testing suite for a single user.

-  `log/plotter.py` \
Plots all of our data from the user test data.

### Games
#### Main Usage:
-  `sgt.py` \
Runs the full suite of games, starting with the game selection menu.

#### Individual Parts:
-  `dots.py` \
Runs the user testing game without any of the data collection.
![dots!](https://github.com/joshdoc/smart_game_table/blob/master/Media/gifs/dots.gif?raw=true)
 

-  `hockey.py` \
Makes the table into an air hockey emulator detecting our tangible air hockey paddles.
![hockey!](https://github.com/joshdoc/smart_game_table/blob/master/Media/gifs/hockey.gif?raw=true)
-  `macrodata.py` \
Plays a game similar to the macrodata refinement job in Severance.
![mysterious and important gif](https://github.com/joshdoc/smart_game_table/blob/master/Media/gifs/mdr.gif?raw=true)

-  `mouse.py` \
Makes the table into a functional touchscreen.
![mouse emulation!](https://github.com/joshdoc/smart_game_table/blob/master/Media/gifs/mousemode.gif?raw=true)
-  `undertable.py` \
Plays a game similar to an Undertale boss fight on the table.
![enter image description here](https://github.com/joshdoc/smart_game_table/blob/master/Media/gifs/undertable.gif?raw=true)
-  `menu.py` \
Runs the menu for switching between games.

### Backend

-  `cv.py` \
Run on its own to see detected centroids. This program is used by the games and
user tests to detect finger presses and tangible object locations.
![enter image description here](https://github.com/joshdoc/smart_game_table/blob/master/Media/gifs/cv.gif?raw=true)

-  `sgt_types.py` \
Contains all data types shared between multiple files.

-  `debug/` \
Provides a script to determine the proper camera orientation and table outline thresholds. (`vis.py`)

-  `graphics/` \
Contains all of the required graphics for the games.