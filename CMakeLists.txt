# Windows compilation hasn't really been tested.


cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -std=c++0x -ffast-math")

# You shouldn't have to modify anything below this line 
########################################################

#find_package(PkgConfig REQUIRED)
#find_package( OpenCV REQUIRED )

INCLUDE_DIRECTORIES(
	./dlib-19.4-modified
)

LINK_DIRECTORIES(
	./dlib-19.4-modified/build/dlib
)

LINK_LIBRARIES(
    dlib
)

ADD_EXECUTABLE(learnRobot
	./frameUtils.cpp
    ./learnRobot.cpp
)